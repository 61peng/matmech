import json
import re
import os
from pathlib import Path
from typing import Optional

def jsonl_to_json(input_path: str, output_dir: Optional[str] = None, overwrite: bool = False):
    """把 input_path 指定的 .jsonl 文件中每一行的 content 字段解析成 JSON，
    并将所有解析后的记录写入到单个 JSON 文件中。

    输出行为：
    - 输出文件与输入文件同名，但扩展名为 .json，存放在 `.../<journal>/json/` 目录下。
    - 如果解析 content 失败，会把原始 content 字段保留，并在输出中加入 _content_parse_failed: true
    """
    inp = Path(input_path)
    if not inp.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 输出文件路径与输入文件同名，但后缀和目录不同
    out_path = out_dir / inp.with_suffix('.json').name
    if out_path.exists() and not overwrite:
        print(f"输出文件已存在且未启用覆盖，将不会执行操作: {out_path}")
        return

    all_records = []
    failed_lines = 0

    with inp.open('r', encoding='utf-8') as fh:
        for i, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            content_raw = record.get('content')
            try:
                parsed = json.loads(content_raw)
            except Exception as e:
                try:
                    first = line.find('{')
                    last = line.rfind('}')
                    if first != -1 and last != -1 and last > first:
                        parsed = json.loads(content_raw[first:last+1])
                    else:
                        raise
                except Exception:
                    print(f"跳过行 {i}：无法解析该行为 JSON - {e}")
                    failed_lines += 1
                    parsed = None
            if parsed is not None:
                record['content'] = parsed
                record['_content_parse_failed'] = False
            else:
                record['_content_parse_failed'] = True

            all_records.append(record)

    with out_path.open('w', encoding='utf-8') as ofh:
        json.dump(all_records, ofh, ensure_ascii=False, indent=2)

    print(f"完成。共处理 {len(all_records)} 条记录，解析失败/跳过 {failed_lines} 行。")
    return out_path


# 请根据需要修改下面的路径，或把本函数导出到脚本中使用。
if __name__ == '__main__':
    journal = "Progress_in_Materials_Science"
    input_path = f'output_file/{journal}/raw_data/paper_type.jsonl'
    output_dir = f'output_file/{journal}/processed_data/'
    overwrite = True  # 是否覆盖已存在的输出文件
    out = jsonl_to_json(input_path, output_dir, overwrite=overwrite)