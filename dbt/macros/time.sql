{# BTS hhmm fields ("0605", "2400", or 605 once a CSV reader casts them) -> minutes after
   local midnight. 2400 is a valid BTS value meaning midnight at the end of the day. #}
{% macro hhmm_to_minutes(col) -%}
    (try_cast(lpad(cast({{ col }} as varchar), 4, '0')[1:2] as integer) * 60
     + try_cast(lpad(cast({{ col }} as varchar), 4, '0')[3:4] as integer))
{%- endmacro %}

{# Wall-clock timestamp in IANA zone `tz` -> naive UTC timestamp. DuckDB/ICU resolves the
   spring-forward gap with the pre-transition offset and the fall-back overlap with the first
   occurrence (same as Python zoneinfo with fold=0). #}
{% macro local_to_utc(ts, tz) -%}
    timezone('UTC', timezone({{ tz }}, {{ ts }}))
{%- endmacro %}
