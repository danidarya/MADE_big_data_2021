1 Исполнитель с максимальным числом скробблов

SELECT t.artist_lastfm FROM (
    SELECT artist_lastfm, scrobbles_lastfm
    FROM artists
    ORDER BY scrobbles_lastfm DESC
    LIMIT 1 ) t

![Результат](/query_results/result1.png?raw=true)

2. Самый популярный тэг на lastfm

SELECT top_tags.tag FROM(
    SELECT tag, count(*) AS tag_count
    FROM artists LATERAL VIEW explode(tags_lastfm) t AS tag
    GROUP BY tag
    ORDER BY tag_count DESC
    LIMIT 1 )  top_tags

3 запрос
Самые популярные исполнители 10 самых популярных тегов ластфм

WITH tag_table as (
    SELECT artist_lastfm, tag, scrobbles_lastfm
    FROM artists LATERAL VIEW explode(tags_lastfm) t AS tag),
top_tag as(
    SELECT tag, count(*) AS tag_count
    FROM tag_table
    GROUP BY tag
    ORDER BY tag_count DESC
    LIMIT 10 )
SELECT DISTINCT artist_lastfm,scrobbles_lastfm
FROM tag_table
WHERE tag  in (
    SELECT tag from top_tag)
ORDER BY scrobbles_lastfm DESC
LIMIT 10



4 запрос
Самые популярные (по числу слушаталей) исполнители из Франции

SELECT artist_lastfm, MAX(listeners_lastfm) AS max_listeners
FROM artists LATERAL VIEW EXPLODE(country_lastfm) c AS country
WHERE country='France'
GROUP BY artist_lastfm
ORDER BY max_listeners DESC
LIMIT 10
