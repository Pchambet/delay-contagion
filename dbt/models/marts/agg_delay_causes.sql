-- Delay-cause decomposition by carrier and local departure hour: how reactionary
-- (late-aircraft) minutes build up through the operating day.

select
    carrier,
    dep_hour_local,
    count(*)                                                        as flights,
    sum(coalesce(carrier_delay, 0))                                 as carrier_min,
    sum(coalesce(weather_delay, 0))                                 as weather_min,
    sum(coalesce(nas_delay, 0))                                     as nas_min,
    sum(coalesce(security_delay, 0))                                as security_min,
    sum(coalesce(late_aircraft_delay, 0))                           as late_aircraft_min,
    sum(coalesce(carrier_delay, 0) + coalesce(weather_delay, 0) + coalesce(nas_delay, 0)
        + coalesce(security_delay, 0) + coalesce(late_aircraft_delay, 0)) as total_cause_min
from {{ ref('stg_flights') }}
where not is_cancelled
group by all
