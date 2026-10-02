{# BTS hhmm fields ("0605", "2400", or 605 once a CSV reader casts them) -> minutes after
   local midnight. 2400 is a valid BTS value meaning midnight at the end of the day. #}
{% macro hhmm_to_minutes(col) -%}
    (try_cast(lpad(cast({{ col }} as varchar), 4, '0')[1:2] as integer) * 60
     + try_cast(lpad(cast({{ col }} as varchar), 4, '0')[3:4] as integer))
{%- endmacro %}

{# Wall-clock timestamp in IANA zone `tz` -> naive UTC timestamp, with Python zoneinfo's
   fold=0 semantics. DuckDB/ICU resolves the spring-forward gap with the pre-transition
   offset (what we want) but the fall-back overlap with the *second* occurrence. Reading the
   time one hour earlier and adding the hour back yields the first occurrence inside the
   overlap and the same instant everywhere else, except just after spring-forward, where it
   is one hour late; the earlier of the two candidates is therefore always the fold=0 answer
   for the 1-hour DST shifts of every US zone. Tested on both transition nights. #}
{% macro local_to_utc(ts, tz) -%}
    least(
        timezone('UTC', timezone({{ tz }}, {{ ts }})),
        timezone('UTC', timezone({{ tz }}, {{ ts }} - interval 1 hour)) + interval 1 hour
    )
{%- endmacro %}
