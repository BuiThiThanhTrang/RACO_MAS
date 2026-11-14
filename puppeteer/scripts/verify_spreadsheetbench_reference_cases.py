"""Verify instruction-derived reference transforms against upstream OJ ranges.

This script is a packet-construction witness.  Reference implementations are
never copied into packet public/private artifacts or model context.
"""
from __future__ import annotations

import argparse
from copy import copy
from datetime import date, datetime
import io
from pathlib import Path
import re
import tarfile

from openpyxl import load_workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils.cell import range_boundaries


TARGETS = {
    "183-8": "'Sheet1'!J3:L6",
    "33722": "'Sheet1'!E2:E6",
    "37456": "'Sample Data'!G12:J15",
    "45738": "'Sheet1'!E6:AB25",
    "13-1": "'LISTS'!A3:D32",
    "267-21": "'merging'!C2:D11",
    "531-18": "'Transpose'!A3:C970",
    "341-14": "'result'!A1:B30",
    "58484": "'Sheet1'!H5:H26",
    "58032": "'Sheet1'!A2:A35",
    "250-20": "'RNM'!A1:J20",
    "398-14": "'COLLECTION'!A2:G9",
    "41-47": "'OUT CAS'!A2:C1529,'OUT CAS'!E2:G586,'OUT CAS'!I2:K13,'OUT CAS'!L2:O8",
    "84-40": "'LIST'!C2:E5",
    "120-24": "'Sheet1'!BN2:BN45",
    "227-40": "'Sheet1'!A14:E17",
    "493-5": "'Imported Data'!A1:F10",
    "585-41": "'Sheet1'!A1:H1",
    "47827": "'3 straights to 3 boxed'!G2:G16",
    "50631": "'Sheet1'!H7:H38",
}


def ranges(value: str):
    for part in value.split(","):
        sheet, cell_range = part.split("!", 1)
        yield sheet.strip().strip("'"), cell_range.strip().strip("'")


def number(value):
    return value if isinstance(value, (int, float)) else 0


def comparable(value):
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, str):
        for pattern in ("%m/%d/%Y", "%Y-%m-%d"):
            try:
                return datetime.strptime(value.strip(), pattern).date().isoformat()
            except ValueError:
                pass
    return value


def formula_text(value):
    return value.text if hasattr(value, "text") else value


def custom_day_value(workbook, row, column):
    sheet = workbook["Sample Data"]
    product = sheet.cell(11, column).value
    start_date, end_date = sheet.cell(row, 6).value, sheet.cell(row + 1, 6).value
    start_time, end_time = sheet["F1"].value, sheet["F2"].value
    def combine(day, clock):
        day_value = day.date() if isinstance(day, datetime) else day
        clock_value = clock.time() if isinstance(clock, datetime) else clock
        return datetime.combine(day_value, clock_value)
    start, end = combine(start_date, start_time), combine(end_date, end_time)
    return sum(
        number(sheet.cell(item, 4).value)
        for item in range(4, sheet.max_row + 1)
        if sheet.cell(item, 3).value == product
        and start <= combine(sheet.cell(item, 1).value, sheet.cell(item, 2).value) <= end
    )


def solve_250_20(workbook):
    sheet = workbook["RNM"]
    last_row = max(row for row in range(2, sheet.max_row + 1) if any(sheet.cell(row, column).value is not None for column in range(1, 12)))
    first_rows, totals, duplicates = {}, {}, []
    for row in range(2, last_row + 1):
        key = (sheet.cell(row, 2).value, sheet.cell(row, 3).value)
        if key not in first_rows:
            first_rows[key] = row
            totals[key] = number(sheet.cell(row, 10).value)
        else:
            totals[key] += number(sheet.cell(row, 10).value)
            duplicates.append(row)
    for key, row in first_rows.items():
        sheet.cell(row, 10).value = totals[key]
    for row in reversed(duplicates):
        sheet.delete_rows(row, 1)


def solve_398_14(workbook):
    output = workbook["COLLECTION"]
    order, records = [], {}
    for row in range(2, output.max_row + 1):
        key = (output.cell(row, 3).value, output.cell(row, 4).value)
        if all(item is not None for item in key) and key not in records:
            order.append(key)
            records[key] = {"br": output.cell(row, 2).value, "sale": 0, "ret": 0}
    for sheet in workbook.worksheets:
        if sheet.title == "COLLECTION":
            continue
        headers = {str(sheet.cell(1, column).value).strip().upper(): column for column in range(1, sheet.max_column + 1) if sheet.cell(1, column).value is not None}
        for row in range(2, sheet.max_row + 1):
            ty, order_code = sheet.cell(row, headers["TY"]).value, sheet.cell(row, headers["OR"]).value
            if ty is None or order_code is None:
                continue
            key = (ty, order_code)
            if key not in records:
                order.append(key)
                records[key] = {"br": sheet.cell(row, headers["BR"]).value, "sale": 0, "ret": 0}
            if "SALE" in headers:
                records[key]["sale"] += number(sheet.cell(row, headers["SALE"]).value)
            if "RET" in headers:
                records[key]["ret"] += number(sheet.cell(row, headers["RET"]).value)
    for row in range(2, output.max_row + 1):
        for column in range(1, 8):
            output.cell(row, column).value = None
    for output_row, key in enumerate(order, 2):
        record = records[key]
        values = [output_row - 1, record["br"], key[0], key[1], record["sale"], record["ret"], record["sale"] - record["ret"]]
        for column, value in enumerate(values, 1):
            cell = output.cell(output_row, column)
            cell.value = value
            cell.font = Font(name="Calibri", size=11, bold=False)
            cell.alignment = Alignment(horizontal="right" if column in (1, 5, 6, 7) else "left")
            if column in (5, 6, 7):
                cell.number_format = "#,##0;[Red]\\-#,##0;\\-"
    for column in range(1, 8):
        output.cell(1, column).font = Font(name="Calibri", size=11, bold=False)
        output.cell(1, column).alignment = Alignment(horizontal="left")


def solve_41_47(workbook):
    output = workbook["OUT CAS"]
    groups = (("PUR PURCHASE", 1), ("PUR PAID", 5), ("CASH SS", 9), ("EXPRR", 13))
    grouped = {name: {} for name, _ in groups}
    for sheet in workbook.worksheets:
        if sheet.title == "OUT CAS":
            continue
        for row in range(2, sheet.max_row + 1):
            reference = sheet.cell(row, 2).value
            if not isinstance(reference, str):
                continue
            for name, _ in groups:
                if reference.startswith(name):
                    grouped[name][reference] = grouped[name].get(reference, 0) + number(sheet.cell(row, 3).value) + number(sheet.cell(row, 4).value)
                    break

    def sort_key(reference):
        match = re.search(r"(\d+)$", reference)
        return (int(match.group(1)) if match else 10**30, reference)

    for name, start_column in groups:
        for row in range(2, output.max_row + 1):
            for column in range(start_column, start_column + 3):
                output.cell(row, column).value = None
        items = sorted(grouped[name].items(), key=lambda item: sort_key(item[0]))
        for output_row, (reference, amount) in enumerate(items, 2):
            output.cell(output_row, start_column).value = output_row - 1
            output.cell(output_row, start_column + 1).value = reference
            output.cell(output_row, start_column + 2).value = amount
        total_row = 2 + len(items)
        output.cell(total_row, start_column).value = "TOTAL"
        output.cell(total_row, start_column + 2).value = sum(amount for _, amount in items)


def solve_84_40(workbook):
    output = workbook["LIST"]
    inputs = [sheet for sheet in workbook.worksheets if sheet.title != "LIST"]
    for row in range(2, output.max_row + 1):
        for column in range(1, 6):
            output.cell(row, column).value = None
    for output_row, sheet in enumerate(inputs, 2):
        imported = sum(number(sheet.cell(row, 3).value) for row in range(2, sheet.max_row + 1))
        exported = sum(number(sheet.cell(row, 4).value) for row in range(2, sheet.max_row + 1))
        for column, value in enumerate((output_row - 1, sheet.title, imported, exported, imported - exported), 1):
            output.cell(output_row, column).value = value
    total_row = 2 + len(inputs)
    output.cell(total_row, 2).value = "TOTAL"
    for column in (3, 4, 5):
        output.cell(total_row, column).value = sum(number(output.cell(row, column).value) for row in range(2, total_row))


def solve_120_24(workbook):
    sheet = workbook["Sheet1"]
    first = "OCOGS - Spares - Transfer Price Overhead"
    second = "OFSS - ORCL Consulting Prod Cost I/C"
    blue = PatternFill(fill_type="solid", fgColor="0070C0")
    for row in range(2, sheet.max_row + 1):
        if sheet.cell(row, 64).value == "LAG" and sheet.cell(row, 59).value in (first, second):
            sheet.cell(row, 66).value = first if sheet.cell(row, 51).value in ("BOA_033E", "BOA_011G") else second
            sheet.cell(row, 64).fill = blue


def solve_227_40(workbook):
    sheet = workbook["Sheet1"]
    output_start = 14
    for row in range(output_start, output_start + 4):
        for column in range(1, 6):
            sheet.cell(row, column).value = None
    for column in range(1, 6):
        sheet.cell(output_start, column).value = sheet.cell(1, column).value
        sheet.cell(output_start, column).font = copy(sheet.cell(1, column).font)
    for source_row in range(2, 5):
        output_row = output_start + source_row - 1
        sheet.cell(output_row, 1).value = sheet.cell(source_row, 1).value
        seen = set()
        for column in range(2, 6):
            value = sheet.cell(source_row, column).value
            if value is not None and value not in seen:
                seen.add(value)
                sheet.cell(output_row, column).value = value


def solve_493_5(workbook):
    sheet = workbook["Imported Data"]
    last_row = max(row for row in range(2, sheet.max_row + 1) if any(sheet.cell(row, column).value is not None for column in range(1, 7)))
    removed = set()
    for left in range(2, last_row + 1):
        if left in removed:
            continue
        for right in range(left + 1, last_row + 1):
            if right in removed:
                continue
            same_reference = sheet.cell(left, 1).value == sheet.cell(right, 1).value
            same_narrative = sheet.cell(left, 6).value == sheet.cell(right, 6).value
            reversed_amounts = (
                sheet.cell(left, 3).value == sheet.cell(right, 4).value
                and sheet.cell(left, 4).value == sheet.cell(right, 3).value
            )
            if same_reference and same_narrative and reversed_amounts:
                removed.update((left, right))
                break
    for row in sorted(removed, reverse=True):
        sheet.delete_rows(row, 1)
    remaining_last = last_row - len(removed)
    for row in range(2, remaining_last + 1):
        sheet.cell(row, 5).value = f"=+C{row}-D{row}"


def solve_585_41(workbook):
    sheet = workbook["Sheet1"]
    for column in range(1, 9):
        cell = sheet.cell(1, column)
        if not isinstance(cell.value, str):
            continue
        cleaned = re.sub(r"[^0-9.,]", "", cell.value).replace(",", "")
        if cleaned:
            cell.value = float(cleaned) if "." in cleaned else int(cleaned)


def solve_47827(workbook):
    sheet = workbook["3 straights to 3 boxed"]
    fill = PatternFill(fill_type="solid", fgColor="EBF1DE")
    for row in range(2, 17):
        digits = f"{int(sheet.cell(row, 5).value):03d}"
        cell = sheet.cell(row, 7)
        cell.value = "".join(sorted(digits))
        font = copy(cell.font)
        font.bold = True
        cell.font = font
        cell.fill = fill
        cell.number_format = "000"


def solve_50631(workbook):
    sheet = workbook["Sheet1"]
    for row in range(7, 39):
        workday = f"WORKDAY($B$3-1,ROWS($B$7:$B{row}),$J$5:$J$7)"
        sheet.cell(row, 8).value = f'=IF({workday}<=$F$3,{workday},"")'


def solve_183_8(workbook):
    sheet = workbook["Sheet1"]
    last_row = max(row for row in range(2, sheet.max_row + 1) if any(sheet.cell(row, column).value is not None for column in range(2, 7)))
    for row in range(3, 7):
        group = sheet.cell(row, 8).value
        for output_column, value_column in zip((10, 11, 12), (4, 5, 6)):
            numerator = sum(number(sheet.cell(item, 3).value) * number(sheet.cell(item, value_column).value) for item in range(2, last_row + 1) if sheet.cell(item, 2).value == group)
            denominator = sum(number(sheet.cell(item, 3).value) for item in range(2, last_row + 1) if sheet.cell(item, 2).value == group)
            sheet.cell(row, output_column).value = numerator / denominator


def solve_183_8_formulas(workbook):
    sheet = workbook["Sheet1"]
    last_row = max(row for row in range(2, sheet.max_row + 1) if any(sheet.cell(row, column).value is not None for column in range(2, 7)))
    for row in range(3, 7):
        for output_column, value_letter in zip((10, 11, 12), ("D", "E", "F")):
            sheet.cell(row, output_column).value = f'=SUMPRODUCT(($B$2:$B${last_row}=$H{row})*$C$2:$C${last_row}*${value_letter}$2:${value_letter}${last_row})/SUMIF($B$2:$B${last_row},$H{row},$C$2:$C${last_row})'


def solve_33722(workbook):
    sheet = workbook["Sheet1"]
    last_row = max(row for row in range(2, sheet.max_row + 1) if sheet.cell(row, 2).value is not None or sheet.cell(row, 3).value is not None)
    thin = Side(style="thin", color="000000")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    for row in range(2, 7):
        cell = sheet.cell(row, 5)
        cell.value = f'=MEDIAN(_xlfn._xlws.FILTER(B{row}:$B${last_row},(C{row}:$C${last_row}>0%)*(C{row}:$C${last_row}<1%)))'
        cell.border = border
        cell.alignment = Alignment(horizontal="left")
        cell.number_format = "0"


def solve_37456(workbook):
    sheet = workbook["Sample Data"]
    last_row = max(row for row in range(4, sheet.max_row + 1) if any(sheet.cell(row, column).value is not None for column in range(1, 5)))
    for row in range(12, 16):
        for column in range(7, 11):
            letter = chr(64 + column)
            sheet.cell(row, column).value = f"=SUMPRODUCT((($C$4:$C${last_row}={letter}$11)*('Sample Data'!$A$4:$A${last_row}+'Sample Data'!$B$4:$B${last_row})>='Sample Data'!$F{row}+$F$1)*'Sample Data'!$D$4:$D${last_row})-SUMPRODUCT((($C$4:$C${last_row}={letter}$11)*('Sample Data'!$A$4:$A${last_row}+'Sample Data'!$B$4:$B${last_row})>'Sample Data'!$F{row + 1}+$F$2)*'Sample Data'!$D$4:$D${last_row})"
            sheet.cell(row, column).number_format = '"$"#,##0.00;[Red]-"$"#,##0.00'


def solve_45738(workbook):
    sheet = workbook["Sheet1"]
    def letter(column):
        result = ""
        while column:
            column, remainder = divmod(column - 1, 26)
            result = chr(65 + remainder) + result
        return result
    for row in range(6, 26):
        for column in range(5, 29):
            name = letter(column)
            cell = sheet.cell(row, column)
            cell.value = f'=(MOD(YEARFRAC({name}$4,EOMONTH($D{row},0))*12,12/$C{row})=0)*(EDATE($D{row},1)>{name}$4)*$B{row}+(EOMONTH($D{row},0)={name}$4)*$A{row}'
            cell.alignment = Alignment(horizontal="center")


def solve_13_1(workbook):
    source, output = workbook["RANGES"], workbook["LISTS"]
    sections = {}
    for header_row in range(1, source.max_row + 1):
        if [source.cell(header_row, column).value for column in range(1, 6)] != ["S.N", "DATE", "BATCH", "REF", "AMOUNTS"]:
            continue
        title, order, totals = str(source.cell(header_row - 1, 3).value), [], {}
        row = header_row + 1
        while row <= source.max_row and source.cell(row, 1).value is not None:
            key = (str(source.cell(row, 2).value), source.cell(row, 4).value)
            if key not in totals:
                order.append((source.cell(row, 2).value, source.cell(row, 4).value, key))
                totals[key] = 0
            totals[key] += number(source.cell(row, 5).value)
            row += 1
        sections[title] = [[index, date_value, reference, totals[key]] for index, (date_value, reference, key) in enumerate(order, 1)]
    for header_row in range(1, output.max_row + 1):
        if [output.cell(header_row, column).value for column in range(1, 5)] != ["SN", "DATE", "REF", "AMOUNTS"]:
            continue
        values = sections[str(output.cell(header_row - 1, 3).value)]
        next_header = min([row for row in range(header_row + 1, output.max_row + 1) if [output.cell(row, column).value for column in range(1, 5)] == ["SN", "DATE", "REF", "AMOUNTS"]] or [output.max_row + 1])
        for row in range(header_row + 1, next_header - 1):
            for column in range(1, 5):
                output.cell(row, column).value = None
        for offset, row_values in enumerate(values, 1):
            for column, value in enumerate(row_values, 1):
                output.cell(header_row + offset, column).value = value
        total_row = header_row + 1 + len(values)
        output.cell(total_row, 1).value = "TOTAL"
        output.cell(total_row, 4).value = f"=SUM(D{header_row + 1}:D{total_row - 1})"


def solve_13_1_sorted(workbook):
    solve_13_1(workbook)
    output = workbook["LISTS"]
    def date_key(value):
        if isinstance(value, str):
            month, day, year = [int(item) for item in value.split("/")]
            return year, month, day
        return value.year, value.month, value.day
    for header_row in range(1, output.max_row + 1):
        if [output.cell(header_row, column).value for column in range(1, 5)] != ["SN", "DATE", "REF", "AMOUNTS"]:
            continue
        row = header_row + 1
        values = []
        while row <= output.max_row and output.cell(row, 1).value != "TOTAL":
            values.append([output.cell(row, column).value for column in range(1, 5)])
            row += 1
        values.sort(key=lambda item: (date_key(item[1]), str(item[2])))
        for offset, row_values in enumerate(values, 1):
            row_values[0] = offset
            for column, value in enumerate(row_values, 1):
                output.cell(header_row + offset, column).value = value


def solve_267_21(workbook):
    output = workbook["merging"]
    for output_column, source_name in ((3, "SH1"), (4, "RP1")):
        source = workbook[source_name]
        headers = {str(source.cell(1, column).value).strip().upper(): column for column in range(1, source.max_column + 1) if source.cell(1, column).value is not None}
        lookup = {source.cell(row, headers["ID"]).value: source.cell(row, headers["QTY"]).value for row in range(2, source.max_row + 1)}
        for row in range(2, 12):
            value = lookup.get(output.cell(row, 2).value)
            output.cell(row, output_column).value = "-" if value in (None, "") else value


def solve_531_18(workbook):
    source, output = workbook["Data"], workbook["Transpose"]
    last_column = max(column for column in range(7, source.max_column + 1) if source.cell(6, column).value is not None)
    last_row = max(row for row in range(7, source.max_row + 1) if source.cell(row, 1).value is not None)
    for row in range(3, output.max_row + 1):
        for column in range(1, 4):
            output.cell(row, column).value = None
    output_row = 3
    for row in range(7, last_row + 1):
        for column in range(7, last_column + 1):
            output.cell(output_row, 1).value = source.cell(row, 1).value
            output.cell(output_row, 2).value = source.cell(6, column).value
            output.cell(output_row, 3).value = source.cell(row, column).value
            output_row += 1


def solve_341_14(workbook):
    source, output = workbook["problem"], workbook["result"]
    for row in range(1, output.max_row + 1):
        output.cell(row, 1).value = None
        output.cell(row, 2).value = None
    output_row = 1
    for row in range(1, source.max_row + 1):
        key, raw = source.cell(row, 1).value, source.cell(row, 2).value
        if key is None and raw is None:
            continue
        tokens = str(raw).splitlines() if raw not in (None, "") else [None]
        for token in tokens:
            output.cell(output_row, 1).value = key
            if isinstance(token, str):
                token = token.strip()
                value = int(token) if re.fullmatch(r"[-+]?\d+", token) else float(token) if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", token) else token
            else:
                value = token
            output.cell(output_row, 2).value = value
            output_row += 1


def solve_58484(workbook):
    sheet = workbook["Sheet1"]
    for row in range(5, 27):
        sheet.cell(row, 8).value = f'=IF(OR(D{row}<>5551234,D{row + 1}=5551234),"",COUNTIF($D$5:D{row},5551234)-SUM($H$4:H{row - 1}))'


def solve_58032(workbook):
    sheet = workbook["Sheet1"]
    fill = PatternFill(fill_type="solid", fgColor="FCD5B4")
    last_row = max(row for row in range(3, sheet.max_row + 1) if any(sheet.cell(row, column).value is not None for column in (8, 9, 10)))
    for row in range(2, 36):
        sheet.cell(row, 1).value = f'=IFERROR(INDEX($H$3:$H${last_row},MATCH(1,INDEX(($J$3:$J${last_row}=C{row})*(I$3:$I${last_row}=B{row}),0),0)),"")'
        sheet.cell(row, 1).fill = fill
    sheet.sheet_view.showGridLines = False


SOLVERS = {
    "183-8": solve_183_8,
    "33722": solve_33722,
    "37456": solve_37456,
    "45738": solve_45738,
    "13-1": solve_13_1,
    "267-21": solve_267_21,
    "531-18": solve_531_18,
    "341-14": solve_341_14,
    "58484": solve_58484,
    "58032": solve_58032,
    "250-20": solve_250_20,
    "398-14": solve_398_14,
    "41-47": solve_41_47,
    "84-40": solve_84_40,
    "120-24": solve_120_24,
    "227-40": solve_227_40,
    "493-5": solve_493_5,
    "585-41": solve_585_41,
    "47827": solve_47827,
    "50631": solve_50631,
}


PREFLIGHT_SOLVERS = {**SOLVERS, "183-8": solve_183_8_formulas, "13-1": solve_13_1_sorted}


def compare(generated, generated_formulas, expected, expected_formulas, target):
    mismatches = []
    for sheet, cell_range in ranges(target):
        min_col, min_row, max_col, max_row = range_boundaries(cell_range)
        for row in range(min_row, max_row + 1):
            for column in range(min_col, max_col + 1):
                actual = generated[sheet].cell(row, column).value
                wanted = expected[sheet].cell(row, column).value
                actual_formula = generated_formulas[sheet].cell(row, column).value
                formula = expected_formulas[sheet].cell(row, column).value
                actual_formula, formula = formula_text(actual_formula), formula_text(formula)
                if isinstance(actual_formula, str) and actual_formula.startswith("=") and isinstance(formula, str) and formula.startswith("="):
                    if re.sub(r"\s+", "", actual_formula).upper() == re.sub(r"\s+", "", formula).upper():
                        continue
                if sheet == "Sample Data" and isinstance(actual_formula, str) and actual_formula.startswith("=SUMPRODUCT") and isinstance(wanted, (int, float)):
                    actual = custom_day_value(generated_formulas, row, column)
                if isinstance(actual_formula, str) and isinstance(wanted, (int, float)):
                    match = re.fullmatch(r"=SUM\(([A-Z]+)(\d+):([A-Z]+)(\d+)\)", actual_formula, re.I)
                    if match and match.group(1).upper() == match.group(3).upper():
                        actual = sum(number(generated_formulas[sheet][f"{match.group(1)}{item}"].value) for item in range(int(match.group(2)), int(match.group(4)) + 1))
                if wanted is None and isinstance(formula, str):
                    match = re.fullmatch(r"=SUM\(([A-Z]+)(\d+):([A-Z]+)(\d+)\)", formula, re.I)
                    if match and match.group(1).upper() == match.group(3).upper():
                        wanted = sum(number(expected_formulas[sheet][f"{match.group(1)}{item}"].value) for item in range(int(match.group(2)), int(match.group(4)) + 1))
                equal = comparable(actual) == comparable(wanted)
                if isinstance(actual, (int, float)) and isinstance(wanted, (int, float)):
                    equal = round(float(actual), 2) == round(float(wanted), 2)
                if not equal:
                    mismatches.append({"sheet": sheet, "cell": generated[sheet].cell(row, column).coordinate, "actual": actual, "expected": wanted})
                    if len(mismatches) >= 20:
                        return mismatches
    return mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--source-id", action="append", choices=sorted(SOLVERS))
    args = parser.parse_args()
    source_ids = args.source_id or list(SOLVERS)
    all_passed = True
    with tarfile.open(args.archive, "r:gz") as archive:
        for source_id in source_ids:
            print(source_id)
            for case_id in (1, 2, 3):
                prefix = f"all_data_912_v0.1/spreadsheet/{source_id}/{case_id}_{source_id}_"
                generated = load_workbook(io.BytesIO(archive.extractfile(prefix + "input.xlsx").read()), data_only=False)
                SOLVERS[source_id](generated)
                buffer = io.BytesIO()
                generated.save(buffer)
                generated.close()
                actual = load_workbook(io.BytesIO(buffer.getvalue()), data_only=True)
                actual_formulas = load_workbook(io.BytesIO(buffer.getvalue()), data_only=False)
                answer_bytes = archive.extractfile(prefix + "answer.xlsx").read()
                expected = load_workbook(io.BytesIO(answer_bytes), data_only=True)
                expected_formulas = load_workbook(io.BytesIO(answer_bytes), data_only=False)
                mismatches = compare(actual, actual_formulas, expected, expected_formulas, TARGETS[source_id])
                actual.close()
                actual_formulas.close()
                expected.close()
                expected_formulas.close()
                passed = not mismatches
                all_passed &= passed
                print(f"  case {case_id}: {'PASS' if passed else 'FAIL'} {mismatches[:3]}")
    raise SystemExit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
