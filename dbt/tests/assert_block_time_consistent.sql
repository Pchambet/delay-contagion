-- UTC arrival - UTC departure must reproduce the published block time. A residual means a
-- wrong time zone or a mis-resolved arrival date. Small residuals exist in the source
-- (a handful of records per month with an inconsistent CRSElapsedTime); a systematic
-- time-zone bug would produce thousands, so we warn on any and fail beyond 1,000.
{{ config(severity='error', warn_if='>0', error_if='>1000') }}
select flight_id, block_time_residual_min
from {{ ref('stg_flights') }}
where abs(block_time_residual_min) > 30
