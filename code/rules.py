"""Three black rules on every table (copied from DIC抗凝_族群嚴重度/analysis/make_exhibits.py)."""

def apply_three_rules(path):
    """補上三線表的黑線。

    docbuild 產出的表格完全沒有框線，投稿慣例是三條黑線：表頭上緣、表頭下緣、
    表格最末列下緣，沒有直線也沒有其他橫線。格式對照使用者桌面的
    `tables and figures.docx`（嚴重氣喘專案交付版）：single、sz = 8、color 000000，
    且表頭下一列補一條上緣線與表頭下緣線重合。

    圖說列不畫線。跨頁表的圖說被 docbuild 併進表格首列（單一 tc、gridSpan 全寬），
    這種表的表頭是第二列，要跳過首列。共用的 docbuild.py 不動。
    """
    from docx import Document
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    def borders(*edges):
        b = OxmlElement("w:tcBorders")
        for e in edges:
            el = OxmlElement(f"w:{e}")
            el.set(qn("w:val"), "single")
            el.set(qn("w:sz"), "8")
            el.set(qn("w:space"), "0")
            el.set(qn("w:color"), "000000")
            b.append(el)
        return b

    def rule(tr, *edges):
        for tc in tr.findall(qn("w:tc")):
            tcPr = tc.find(qn("w:tcPr"))
            if tcPr is None:
                tcPr = OxmlElement("w:tcPr")
                tc.insert(0, tcPr)
            for old in tcPr.findall(qn("w:tcBorders")):
                tcPr.remove(old)
            tcPr.append(borders(*edges))   # tcW → gridSpan → tcBorders，順序合法

    d = Document(str(path))
    n = 0
    for tbl in d.element.body.findall(qn("w:tbl")):
        rows = tbl.findall(qn("w:tr"))
        if not rows:
            continue
        # 首列若只有一個儲存格而次列不只一個，那一列是被併進來的圖說
        h = 1 if (len(rows) > 1
                  and len(rows[0].findall(qn("w:tc"))) == 1
                  and len(rows[1].findall(qn("w:tc"))) > 1) else 0
        last = len(rows) - 1
        rule(rows[h], "top", "bottom")
        if h + 1 <= last:
            rule(rows[h + 1], *(("top", "bottom") if h + 1 == last else ("top",)))
        if last > h + 1:
            rule(rows[last], "bottom")
        n += 1
    d.save(str(path))
    return n

