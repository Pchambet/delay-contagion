-- Within a chain, legs are strictly ordered in UTC: each leg departs after the previous
-- one arrives (scheduled), and every leg arrives after it departs.
with legs as (
    select
        chain_id,
        leg_seq,
        crs_dep_utc,
        crs_arr_utc,
        lag(crs_arr_utc) over (partition by chain_id order by leg_seq) as prev_crs_arr_utc
    from {{ ref('fct_legs') }}
)
select *
from legs
where crs_arr_utc <= crs_dep_utc
   or (prev_crs_arr_utc is not null and crs_dep_utc < prev_crs_arr_utc)
