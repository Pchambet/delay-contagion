-- One row per scheduled flight: typed, keyed, with UTC timestamps.
--
-- Scheduled times are published in local time at each airport. Departure is anchored on
-- FlightDate at the origin. The arrival date is not published, so we take the local CRS
-- arrival clock time on the day (-1..+2) that lands closest, in UTC, to departure plus the
-- scheduled block time (CRSElapsedTime). This handles overnight legs, date-line legs
-- (Guam, Saipan) and DST changes without hand-written rules. Actual times are derived as
-- scheduled UTC + signed BTS delay minutes, which avoids the date ambiguity of the raw
-- hhmm actual-time fields.

with source as (
    select * from '{{ var("flights_path") }}'
),

typed as (
    select
        cast(FlightDate as date)                                   as flight_date,
        cast(Reporting_Airline as varchar)                         as carrier,
        nullif(trim(cast(Tail_Number as varchar)), '')             as tail_number,
        cast(Flight_Number_Reporting_Airline as integer)           as flight_number,
        cast(Origin as varchar)                                    as origin,
        cast(Dest as varchar)                                      as dest,
        {{ hhmm_to_minutes('CRSDepTime') }}                        as crs_dep_min,
        {{ hhmm_to_minutes('CRSArrTime') }}                        as crs_arr_min,
        cast(DepDelay as double)                                   as dep_delay,
        cast(ArrDelay as double)                                   as arr_delay,
        coalesce(cast(Cancelled as double), 0) = 1                 as is_cancelled,
        coalesce(cast(Diverted as double), 0) = 1                  as is_diverted,
        cast(CRSElapsedTime as double)                             as crs_elapsed_min,
        cast(Distance as double)                                   as distance_mi,
        cast(CarrierDelay as double)                               as carrier_delay,
        cast(WeatherDelay as double)                               as weather_delay,
        cast(NASDelay as double)                                   as nas_delay,
        cast(SecurityDelay as double)                              as security_delay,
        cast(LateAircraftDelay as double)                          as late_aircraft_delay
    from source
),

localized as (
    select
        t.*,
        o.tz                                                       as origin_tz,
        d.tz                                                       as dest_tz,
        t.flight_date + to_minutes(t.crs_dep_min)                  as crs_dep_local,
        {{ local_to_utc('t.flight_date + to_minutes(t.crs_dep_min)', 'o.tz') }} as crs_dep_utc,
        t.flight_date + to_minutes(t.crs_arr_min)                  as crs_arr_local_d0
    from typed t
    left join {{ ref('airports') }} o on t.origin = o.iata
    left join {{ ref('airports') }} d on t.dest = d.iata
),

arrival_day as (
    select
        *,
        coalesce(
            cast(round(date_diff(
                'minute',
                {{ local_to_utc('crs_arr_local_d0', 'dest_tz') }},
                crs_dep_utc + to_minutes(cast(crs_elapsed_min as bigint))
            ) / 1440.0) as integer),
            -- no block time published: fall back to "arrives the next day if earlier"
            case when crs_arr_min < crs_dep_min then 1 else 0 end
        ) as arr_day_shift
    from localized
),

resolved as (
    select
        *,
        crs_arr_local_d0 + to_days(arr_day_shift)                  as crs_arr_local,
        {{ local_to_utc('crs_arr_local_d0 + to_days(arr_day_shift)', 'dest_tz') }} as crs_arr_utc
    from arrival_day
),

keyed as (
    select
        carrier || '-' || flight_number || '-' || origin || '-'
            || strftime(crs_dep_local, '%Y%m%d%H%M')               as flight_id,
        *
    from resolved
)

select
    flight_id,
    flight_date,
    strftime(flight_date, '%Y-%m')                                 as flight_month,
    carrier,
    tail_number,
    flight_number,
    origin,
    dest,
    origin_tz,
    dest_tz,
    crs_dep_local,
    crs_arr_local,
    crs_dep_utc,
    crs_arr_utc,
    hour(crs_dep_local)                                            as dep_hour_local,
    crs_elapsed_min,
    date_diff('minute', crs_dep_utc, crs_arr_utc) - crs_elapsed_min as block_time_residual_min,
    dep_delay,
    arr_delay,
    crs_dep_utc + to_minutes(cast(dep_delay as bigint))            as act_dep_utc,
    crs_arr_utc + to_minutes(cast(arr_delay as bigint))            as act_arr_utc,
    is_cancelled,
    is_diverted,
    distance_mi,
    arr_delay >= 15 and carrier_delay is not null                  as has_cause_codes,
    carrier_delay,
    weather_delay,
    nas_delay,
    security_delay,
    late_aircraft_delay
from keyed
-- The source occasionally repeats a record; keep one per key.
qualify row_number() over (partition by flight_id order by tail_number nulls last) = 1
