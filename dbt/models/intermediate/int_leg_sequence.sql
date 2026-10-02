-- Every operated leg, placed in its aircraft's operating sequence.
--
-- Legs of the same carrier + tail are ordered by scheduled UTC departure. A leg is *linked*
-- to the previous one (a "turn") when the aircraft physically continues: same station,
-- previous leg not diverted, scheduled ground time in [0, max_turn_min], and a non-negative
-- actual ground time (a negative one means the tail number was swapped or misreported).
-- Legs whose UTC block time disagrees with the published one by > 30 min are dropped.
-- Unlinked legs start a new chain; a chain is one aircraft's continuous run of turns.

with legs as (
    select *
    from {{ ref('stg_flights') }}
    where not is_cancelled
      and tail_number is not null
      and crs_dep_utc is not null
      and crs_arr_utc is not null
      and dep_delay is not null
      -- a few source records have a block time inconsistent with their own clock times
      and abs(coalesce(block_time_residual_min, 0)) <= 30
),

ordered as (
    select
        *,
        lag(flight_id)    over w as prev_flight_id,
        lag(dest)         over w as prev_dest,
        lag(crs_arr_utc)  over w as prev_crs_arr_utc,
        lag(arr_delay)    over w as prev_arr_delay,
        lag(is_diverted)  over w as prev_is_diverted
    from legs
    window w as (partition by carrier, tail_number order by crs_dep_utc, flight_id)
),

turns as (
    select
        *,
        date_diff('minute', prev_crs_arr_utc, crs_dep_utc)          as sched_turn_min,
        date_diff('minute', prev_crs_arr_utc, crs_dep_utc)
            + dep_delay - prev_arr_delay                            as actual_turn_min
    from ordered
),

linked as (
    select
        *,
        coalesce(
            prev_dest = origin
            and not prev_is_diverted
            and prev_arr_delay is not null
            and sched_turn_min between 0 and {{ var('max_turn_min') }}
            and actual_turn_min >= 0,
            false
        ) as is_linked
    from turns
),

chained as (
    select
        *,
        sum(case when is_linked then 0 else 1 end) over (
            partition by carrier, tail_number
            order by crs_dep_utc, flight_id
            rows between unbounded preceding and current row
        ) as chain_no
    from linked
)

select
    flight_id,
    carrier,
    tail_number,
    carrier || '-' || tail_number || '-' || chain_no                as chain_id,
    row_number() over c                                             as leg_seq,
    count(*) over (partition by carrier, tail_number, chain_no)     as chain_legs,
    first_value(flight_date) over c                                 as chain_date,
    is_linked,
    case when is_linked then prev_flight_id end                     as prev_flight_id,
    case when is_linked then sched_turn_min end                     as sched_turn_min,
    case when is_linked then actual_turn_min end                    as actual_turn_min,
    case when is_linked then prev_arr_delay end                     as inbound_arr_delay,
    case when is_linked then prev_crs_arr_utc end                   as inbound_crs_arr_utc
from chained
window c as (partition by carrier, tail_number, chain_no order by crs_dep_utc, flight_id)
