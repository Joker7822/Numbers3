# -*- coding: utf-8 -*-
"""
Mizuho銀行 ナンバーズ3 抽せん結果スクレイパ（堅牢版）
- PC/SPどちらのテーブルにも対応（.js-lottery-temp-pc / -sp / フォールバック）
- 中身(textContent)が入るまで待機 + リロード再試行
- 既存CSVに追記（同一「抽せん日」をスキップ）＆日付昇順整列
"""

import csv
import os
import re
import time
from datetime import datetime

import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

URL = "https://www.mizuhobank.co.jp/takarakuji/check/numbers/numbers3/index.html"
CSV_PATH = "numbers3.csv"

REQUIRED_FIELDNAMES = [
    "抽せん日",
    "本数字",
    "回別",
    "ストレート",
    "ボックス",
    "セット(ストレート)",
    "セット(ボックス)",
    "ミニ",
]

TABLE_SELECTOR_ANY = (
    ".section__table-wrap .js-lottery-temp-pc, "
    ".section__table-wrap .js-lottery-temp-sp, "
    ".section__table-wrap table.section__table"
)


def build_driver(headless: bool = True) -> webdriver.Chrome:
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--window-size=1366,2600")
    options.add_argument(
        "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    )
    options.add_argument("--lang=ja-JP,ja")
    options.add_argument("--disable-blink-features=AutomationControlled")
    driver = webdriver.Chrome(options=options)
    return driver


def _find_first(scope_el, selectors):
    for css in selectors:
        try:
            el = scope_el.find_element(By.CSS_SELECTOR, css)
            if el:
                return el
        except Exception:
            continue
    return None


def _filled(el):
    try:
        return el is not None and el.get_attribute("textContent").strip() != ""
    except Exception:
        return False


def wait_for_tables(driver, timeout_total: int = 45, retries: int = 2):
    """
    PC/SPいずれかのテーブルが現れ、先頭テーブルの date/issue/number が埋まるまで待つ。
    必要に応じて再読み込みを数回試行。
    """
    attempt = 0
    while attempt <= retries:
        attempt += 1
        try:
            wait = WebDriverWait(driver, timeout_total)

            try:
                wait.until(EC.invisibility_of_element_located((By.CSS_SELECTOR, ".js-now-loading")))
            except Exception:
                pass

            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, TABLE_SELECTOR_ANY)))

            t_end = time.time() + timeout_total
            while time.time() < t_end:
                tables = driver.find_elements(By.CSS_SELECTOR, TABLE_SELECTOR_ANY)
                tables = [t for t in tables if t.is_displayed()] or tables
                if not tables:
                    time.sleep(0.5)
                    continue

                t0 = tables[0]
                date_el = _find_first(t0, [".js-lottery-date-pc", ".js-lottery-date"])
                issue_el = _find_first(t0, [".js-lottery-issue-pc", ".js-lottery-issue"])
                num_el = _find_first(t0, [".js-lottery-number-pc", ".js-lottery-number"])

                if date_el and issue_el and num_el and all(_filled(x) for x in (date_el, issue_el, num_el)):
                    return
                time.sleep(0.5)

            if attempt <= retries:
                driver.refresh()
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
                time.sleep(1.2)
                continue
            else:
                raise TimeoutError("テーブルが埋まる前にタイムアウトしました。")

        except Exception:
            if attempt <= retries:
                driver.refresh()
                time.sleep(1.0)
                continue
            raise


def parse_tables(driver):
    tables = driver.find_elements(By.CSS_SELECTOR, TABLE_SELECTOR_ANY)
    tables = [t for t in tables if t.is_displayed()] or tables

    results = []
    for t in tables:
        date_el = _find_first(t, [".js-lottery-date-pc", ".js-lottery-date"])
        issue_el = _find_first(t, [".js-lottery-issue-pc", ".js-lottery-issue"])
        num_el = _find_first(t, [".js-lottery-number-pc", ".js-lottery-number"])
        if not (date_el and issue_el and num_el):
            continue

        date_text = date_el.get_attribute("textContent").strip()
        issue_text = issue_el.get_attribute("textContent").strip()
        num_text = num_el.get_attribute("textContent").strip()
        if not (date_text and issue_text and re.search(r"\d", num_text or "")):
            continue

        try:
            draw_date = datetime.strptime(date_text, "%Y年%m月%d日").strftime("%Y-%m-%d")
        except Exception:
            m = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", date_text)
            if not m:
                continue
            y, mo, d = map(int, m.groups())
            draw_date = f"{y:04d}-{mo:02d}-{d:02d}"

        digits = [int(d) for d in re.findall(r"\d", num_text)]
        main_number = str(digits)

        prize_rows = t.find_elements(By.CSS_SELECTOR, "tr.js-lottery-prize-pc, tr.js-lottery-prize")
        prizes = []
        for pr in prize_rows:
            strongs = pr.find_elements(By.CSS_SELECTOR, "td strong.section__text--bold, td strong")
            val = None
            if strongs:
                txt = strongs[-1].get_attribute("textContent").strip()
                if "円" in txt:
                    try:
                        val = int(txt.replace(",", "").replace("円", "").strip())
                    except Exception:
                        val = None
            prizes.append(val)

        while len(prizes) < 5:
            prizes.append(None)

        results.append(
            {
                "回別": issue_text,
                "抽せん日": draw_date,
                "本数字": main_number,
                "ストレート": prizes[0],
                "ボックス": prizes[1],
                "セット(ストレート)": prizes[2],
                "セット(ボックス)": prizes[3],
                "ミニ": prizes[4],
            }
        )
    return results


def load_existing(csv_path: str):
    try:
        df = pd.read_csv(csv_path)
        df["抽せん日"] = df["抽せん日"].astype(str)

        for col in REQUIRED_FIELDNAMES:
            if col not in df.columns:
                df[col] = None

        fieldnames = REQUIRED_FIELDNAMES.copy()
        existing_dates = set(df["抽せん日"].tolist())

        return df, existing_dates, fieldnames

    except FileNotFoundError:
        return pd.DataFrame(), set(), REQUIRED_FIELDNAMES.copy()


def append_and_sort(csv_path: str, fieldnames, new_rows):
    if not new_rows:
        return 0

    normalized_rows = []
    for row in new_rows:
        normalized_rows.append({
            key: row.get(key, "")
            for key in fieldnames
        })

    write_header = (not os.path.exists(csv_path)) or os.path.getsize(csv_path) == 0

    with open(csv_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore"
        )

        if write_header:
            writer.writeheader()

        writer.writerows(normalized_rows)

    df = pd.read_csv(csv_path)

    for col in REQUIRED_FIELDNAMES:
        if col not in df.columns:
            df[col] = None

    df = df[REQUIRED_FIELDNAMES]

    try:
        df["抽せん日_dt"] = pd.to_datetime(df["抽せん日"], errors="coerce")
        df.sort_values(["抽せん日_dt", "回別"], inplace=True)
        df.drop(columns=["抽せん日_dt"], inplace=True)
    except Exception:
        df.sort_values(["抽せん日", "回別"], inplace=True)

    df.to_csv(csv_path, index=False, encoding="utf-8")
    return len(normalized_rows)


def main():
    driver = build_driver(headless=True)
    try:
        driver.get(URL)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/3);")
        time.sleep(0.8)

        wait_for_tables(driver, timeout_total=45, retries=2)
        scraped = parse_tables(driver)
    finally:
        driver.quit()

    if not scraped:
        print("[INFO] 表が取得できませんでした（ページ構造変更/一時的な読込失敗の可能性）。")
        return

    existing_df, existing_dates, fieldnames = load_existing(CSV_PATH)
    new_rows = [row for row in scraped if row["抽せん日"] not in existing_dates]

    saved = append_and_sort(CSV_PATH, fieldnames, new_rows)

    if saved:
        print(f"[INFO] {saved}件を保存し、日付順に並び替えました。CSV: {CSV_PATH}")
    else:
        print("[INFO] 既存CSVと同一日のため、新規保存はありません。")

    for row in new_rows:
        print(row)


if __name__ == "__main__":
    main()
