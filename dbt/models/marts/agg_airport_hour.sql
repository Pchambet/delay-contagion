-- Station x local departure hour: volume, punctuality, and delay-cause minutes.

select
    origin                                                          as airport,
    dep_hour_local,
    count(*)                                                        as flights,
    avg(dep_delay)                                                  as mean_dep_delay,
    avg(case when dep_delay >= 15 then 1.0 else 0.0 end)            as share_dep_delayed_15,
    sum(coalesce(carrier_delay, 0))                                 as carrier_min,
    sum(coalesce(weather_delay, 0))                                 as weather_min,
    sum(coalesce(nas_delay, 0))                                     as nas_min,
    sum(coalesce(security_delay, 0))                                as security_min,
    sum(coalesce(late_aircraft_delay, 0))                           as late_aircraft_min
from {{ ref('stg_flights') }}
where not is_cancelled
group by all
