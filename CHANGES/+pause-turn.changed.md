A message that stopped with `pause_turn` is now `IncompleteReason.PAUSED` (it was `OTHER`; `raw_incomplete_reason` is still `pause_turn`). Send `to_turn()` back to let Anthropic go on with the turn.
