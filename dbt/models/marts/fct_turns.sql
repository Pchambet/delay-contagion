-- One row per aircraft turn, ready for the propagation model:
-- outbound departure delay ~ inbound arrival delay x scheduled slack.

select
    r.turn_id,
    r.in_flight_id,
    r.out_flight_id,
    r.chain_id,
    r.carrier,
    r.tail_number,
    r.station,
    o.flight_date,
    o.flight_month,
    o.dep_hour_local,
    r.sched_turn_min,
    r.actual_turn_min,
    r.inbound_arr_delay,
    r.outbound_dep_delay,
    o.arr_delay                                                     as outbound_arr_delay,
    -- Ground time left once the late inbound is on the gate; propagation starts when this
    -- falls below the effective minimum turn time.
    r.sched_turn_min - greatest(r.inbound_arr_delay, 0)             as available_turn_min,
    o.late_aircraft_delay                                           as outbound_late_aircraft_delay
from {{ ref('int_rotations') }} r
join {{ ref('stg_flights') }} o on o.flight_id = r.out_flight_id
