-- After filtering, every linked turn has non-negative scheduled and actual ground time.
select turn_id, sched_turn_min, actual_turn_min
from {{ ref('fct_turns') }}
where sched_turn_min < 0 or actual_turn_min < 0
