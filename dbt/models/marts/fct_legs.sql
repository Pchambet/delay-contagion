-- Analysis grain for contagion and the buffer LP: every operated leg with its position in
-- the aircraft's chain and the delay it carried in and out.

select
    f.flight_id,
    f.flight_date,
    f.flight_month,
    f.carrier,
    f.tail_number,
    s.chain_id,
    s.chain_date,
    s.leg_seq,
    s.chain_legs,
    s.chain_legs - s.leg_seq                                        as legs_remaining,
    f.origin,
    f.dest,
    f.dep_hour_local,
    case
        when f.dep_hour_local < 6  then 'night'
        when f.dep_hour_local < 12 then 'morning'
        when f.dep_hour_local < 18 then 'afternoon'
        else 'evening'
    end                                                             as dep_period,
    f.crs_dep_utc,
    f.crs_arr_utc,
    s.is_linked,
    s.prev_flight_id,
    s.sched_turn_min,
    s.inbound_arr_delay,
    f.dep_delay,
    f.arr_delay,
    f.is_diverted,
    f.carrier_delay,
    f.weather_delay,
    f.nas_delay,
    f.security_delay,
    f.late_aircraft_delay
from {{ ref('int_leg_sequence') }} s
join {{ ref('stg_flights') }} f using (flight_id)
