-- One row per turn: an inbound leg and the outbound leg the same aircraft flies next.
-- Scheduled turn time = next CRS departure - previous CRS arrival (both UTC).

select
    s.flight_id                                                     as turn_id,
    s.prev_flight_id                                                as in_flight_id,
    s.flight_id                                                     as out_flight_id,
    s.chain_id,
    s.leg_seq                                                       as out_leg_seq,
    s.carrier,
    s.tail_number,
    f.origin                                                        as station,
    s.inbound_crs_arr_utc                                           as in_crs_arr_utc,
    f.crs_dep_utc                                                   as out_crs_dep_utc,
    s.sched_turn_min,
    s.actual_turn_min,
    s.inbound_arr_delay,
    f.dep_delay                                                     as outbound_dep_delay
from {{ ref('int_leg_sequence') }} s
join {{ ref('stg_flights') }} f using (flight_id)
where s.is_linked
