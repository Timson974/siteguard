-- SiteGuard analytics queries (PostgreSQL).
-- Connect: docker compose exec db psql -U siteguard -d siteguard
-- Real model output only: source = 'api'. Synthetic demo rows: source = 'synthetic'.

-- 1. Totals
SELECT COUNT(*) AS frames, SUM(n_persons) AS persons, SUM(n_violations) AS violations,
       ROUND(AVG(latency_ms)::numeric, 1) AS avg_latency_ms
FROM inferences
WHERE source = 'api';

-- 2. Which camera (site zone) has the most violations, normalised by number of frames
SELECT camera_id, COUNT(*) AS frames, SUM(n_violations) AS violations,
       ROUND(SUM(n_violations)::numeric / COUNT(*), 2) AS violations_per_frame
FROM inferences
WHERE source = 'api'
GROUP BY camera_id
ORDER BY violations_per_frame DESC;

-- 3. Violations per day and type
SELECT DATE(i.captured_at) AS day, d.class_name, COUNT(*) AS violations
FROM detections d
JOIN inferences i ON i.id = d.inference_id
WHERE d.is_violation AND i.source = 'api'
GROUP BY 1, 2
ORDER BY 1, 2;

-- 4. Time of day: when during the shift do violations happen?
SELECT EXTRACT(HOUR FROM captured_at) AS hour, SUM(n_violations) AS violations
FROM inferences
WHERE source = 'api'
GROUP BY 1
ORDER BY 1;

-- 5. Confidence of violation detections: how sure is the model?
--    Many low-confidence violations => consider a higher threshold or more training data.
SELECT class_name,
       COUNT(*) AS n,
       ROUND(AVG(confidence)::numeric, 3) AS avg_conf,
       SUM(CASE WHEN confidence < 0.5 THEN 1 ELSE 0 END) AS low_conf
FROM detections
WHERE is_violation
GROUP BY class_name;

-- 6. Clean up demo data
-- DELETE FROM inferences WHERE source = 'synthetic';
