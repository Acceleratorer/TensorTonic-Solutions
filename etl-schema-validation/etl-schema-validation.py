def validate_records(records: list, schema: list) -> list:
      results = []

      for record_index, record in enumerate(records):
          errors = []

          for field in schema:
              column = field["column"]

              if column not in record:
                  errors.append(f"{column}: missing")
                  continue

              value = record[column]

              if value is None:
                  if field["nullable"]:
                      continue
                  errors.append(f"{column}: null")
                  continue

              expected_type = field["type"]

              if expected_type == "int":
                  valid_type = type(value) is int
              elif expected_type == "float":
                  valid_type = type(value) in (int, float)
              elif expected_type == "str":
                  valid_type = type(value) is str
              else:
                  valid_type = False

              if not valid_type:
                  errors.append(
                      f"{column}: expected {expected_type}, got {type(value).__name__}"
                  )
                  continue

              if "min" in field and value < field["min"]:
                  errors.append(f"{column}: out of range")
                  continue

              if "max" in field and value > field["max"]:
                  errors.append(f"{column}: out of range")

          results.append({
              "record_index": record_index,
              "is_valid": not errors,
              "errors": errors,
          })

      return results