"""Metacognition — confidence gating, counterfactual replay, dream cycle."""

import json
import logging
import pickle
from datetime import datetime

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════
# CONFIDENCE GATE
# ══════════════════════════════════════════

async def assess_confidence(provider, user_input: str, response: str,
                            config: dict) -> dict:
    """Quick Haiku self-assessment of response confidence.

    Returns {"confidence": int, "reason": str, "action": str}.
    """
    prompt = (
        "Rate confidence 1-10 for this response. Consider: factual accuracy, "
        "completeness, relevance. Return JSON only:\n"
        '{"confidence":N,"reason":"brief","action":"none|search|escalate|admit"}\n\n'
        f"User: {user_input[:500]}\nResponse: {response[:500]}"
    )
    model = config.get("extraction_model", "claude-haiku-4-5-20251001")
    try:
        result = await provider.complete(
            model=model, max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        text = result.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        return json.loads(text)
    except Exception as e:
        logger.debug("Confidence assessment failed: %s", e)
        return {"confidence": 10, "reason": "assessment_failed", "action": "none"}


def log_interaction(db, user_id: str, user_input: str, response: str,
                    tool_calls: list, success: int, confidence: float | None,
                    model: str):
    """Log interaction for counterfactual replay and proactive analysis."""
    db.execute(
        """INSERT INTO interaction_log
           (user_id, user_input, agent_response, tool_calls_json, success,
            confidence, model_used, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (user_id, user_input[:2000], response[:2000],
         json.dumps(tool_calls), success, confidence, model,
         datetime.now().isoformat()))
    db.commit()


# ══════════════════════════════════════════
# COUNTERFACTUAL REPLAY
# ══════════════════════════════════════════

async def run_counterfactual_replay(provider, db, memory_system,
                                    config: dict) -> int:
    """Replay failures with Haiku, extract lessons. Returns lesson count."""
    max_replays = config.get("max_replays_per_run", 10)
    last_id = memory_system.get_state("app:last_replay_id") or 0

    rows = db.execute(
        """SELECT id, user_id, user_input, agent_response, tool_calls_json
           FROM interaction_log
           WHERE success = 0 AND id > ?
           ORDER BY id LIMIT ?""",
        (last_id, max_replays)).fetchall()

    if not rows:
        return 0

    model = config.get("extraction_model", "claude-haiku-4-5-20251001")
    lessons_count = 0

    for row_id, user_id, user_input, response, tools_json in rows:
        prompt = (
            "Analyze this failed interaction. What went wrong and what should "
            "have been done differently? Return JSON:\n"
            '{"lesson":"concise lesson","category":"tool_error|wrong_approach|missing_info"}\n\n'
            f"User: {user_input[:500]}\nResponse: {response[:500]}\n"
            f"Tools used: {(tools_json or '[]')[:300]}"
        )
        try:
            result = await provider.complete(
                model=model, max_tokens=150,
                messages=[{"role": "user", "content": prompt}])
            text = result.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(text)
            lesson = data.get("lesson", "")
            if lesson and len(lesson) > 10:
                memory_system.remember(
                    f"[lesson] {lesson}", user_id, "lesson", 0.9)
                lessons_count += 1
        except Exception as e:
            logger.warning("Replay failed for id=%d: %s", row_id, e)
        memory_system.set_state("app:last_replay_id", row_id)

    return lessons_count


# ══════════════════════════════════════════
# DREAM CYCLE MEMORY CONSOLIDATION
# ══════════════════════════════════════════

async def run_dream_cycle(provider, db, memory_system, config: dict) -> dict:
    """Consolidate memories during off-hours. Returns stats dict."""
    threshold = config.get("similarity_threshold", 0.85)
    decay_rate = config.get("decay_rate", 0.05)
    max_consolidations = config.get("max_consolidations_per_run", 20)
    model = config.get("extraction_model", "claude-haiku-4-5-20251001")

    stats = {"decayed": 0, "merged": 0, "insights": 0}

    # 1. Decay importance of old un-accessed memories
    cur = db.execute(
        """UPDATE memories SET importance = MAX(importance - ?, 0.05)
           WHERE accessed_at < datetime('now', '-7 days')
           AND importance > 0.1""", (decay_rate,))
    db.commit()
    stats["decayed"] = cur.rowcount

    # 2. Find and merge similar memory clusters per user
    users = [r[0] for r in db.execute(
        "SELECT DISTINCT user_id FROM memories").fetchall()]

    for user_id in users:
        rows = db.execute(
            """SELECT id, content, embedding, importance
               FROM memories WHERE user_id = ? AND embedding IS NOT NULL""",
            (user_id,)).fetchall()

        if len(rows) < 2:
            continue

        clusters = _find_clusters(rows, threshold, memory_system)

        for cluster_ids, cluster_contents in clusters[:max_consolidations]:
            merge_prompt = (
                "Merge these related facts into one concise consolidated fact:\n"
                + "\n".join(f"- {c}" for c in cluster_contents)
                + "\n\nReturn only the merged fact, nothing else."
            )
            try:
                result = await provider.complete(
                    model=model, max_tokens=100,
                    messages=[{"role": "user", "content": merge_prompt}])
                merged = result.content[0].text.strip()
                if merged and len(merged) > 10:
                    max_imp = max(
                        r[3] for r in rows if r[0] in set(cluster_ids))
                    placeholders = ",".join("?" * len(cluster_ids))
                    db.execute(
                        f"DELETE FROM memories WHERE id IN ({placeholders})",
                        cluster_ids)
                    db.commit()
                    memory_system.remember(
                        merged, user_id, "consolidated",
                        min(max_imp + 0.1, 1.0))
                    stats["merged"] += 1
            except Exception as e:
                logger.warning("Dream merge failed: %s", e)

    logger.info("Dream cycle complete: %s", stats)
    return stats


def _find_clusters(rows, threshold: float, memory_system) -> list:
    """Find groups of similar memories by embedding cosine similarity."""
    clusters = []
    used = set()

    for i, (id_a, content_a, emb_a_blob, _) in enumerate(rows):
        if id_a in used:
            continue
        try:
            emb_a = pickle.loads(emb_a_blob)
        except Exception:
            continue

        cluster_ids = [id_a]
        cluster_contents = [content_a]

        for j in range(i + 1, len(rows)):
            id_b, content_b, emb_b_blob, _ = rows[j]
            if id_b in used:
                continue
            try:
                emb_b = pickle.loads(emb_b_blob)
                sim = memory_system._cosine_similarity(emb_a, emb_b)
                if sim >= threshold:
                    cluster_ids.append(id_b)
                    cluster_contents.append(content_b)
                    used.add(id_b)
            except Exception:
                continue

        if len(cluster_ids) > 1:
            used.add(id_a)
            clusters.append((cluster_ids, cluster_contents))

    return clusters
