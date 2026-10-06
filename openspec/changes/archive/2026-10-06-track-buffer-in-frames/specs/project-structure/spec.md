# Spec Delta

## MODIFIED Requirements

### Requirement: Tracker override 方法本體不變
`ClassAwareBOTSORT` 與 `ClassAwareBYTETracker` 中 override 自 ultralytics 的方法（`__init__`、`update`、`remove_duplicate_stracks`，以及 BoTSORT 的 `init_track`）的本體 SHALL 與重構前完全相同；搬移時只允許變更檔案層級的 import 敘述。唯一的例外是兩個 `update` 中把逐類別 `track_buffer` 換算成保留期限的那一個敘述（使用者明確指示，見 ADR 0009）：BoTSORT 版為 `cls_max_time_lost = int(cls_buffer_sec)`，ByteTrack 版為 `cls_max_time_lost = int(cls_buffer)`；除此之外的任何差異都不允許。

#### Scenario: 方法本體比對
- **WHEN** 以語法樹（AST）比對重構前後上述方法的本體，比對前只把重構前版本中這兩個保留期限敘述替換成例外允許的新敘述
- **THEN** 兩者完全相同

#### Scenario: 例外以外的改動被拒絕
- **WHEN** 兩個 `update` 中保留期限敘述以外的任何敘述，或其他凍結方法的本體與重構前不同
- **THEN** 比對失敗
