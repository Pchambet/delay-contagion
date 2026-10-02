-- Carrier x month scorecard with the BTS delay-cause decomposition.

select
    carrier,
    flight_month,
    count(*)                                                        as flights,
    avg(case when is_cancelled then 1.0 else 0.0 end)               as cancel_rate,
    avg(case when not is_cancelled and not is_diverted
             then case when arr_delay < 15 then 1.0 else 0.0 end end) as on_time_rate,
    sum(coalesce(carrier_delay, 0))                                 as carrier_min,
    sum(coalesce(weather_delay, 0))                                 as weather_min,
    sum(coalesce(nas_delay, 0))                                     as nas_min,
    sum(coalesce(security_delay, 0))                                as security_min,
    sum(coalesce(late_aircraft_delay, 0))                           as late_aircraft_min
from {{ ref('stg_flights') }}
group by all
