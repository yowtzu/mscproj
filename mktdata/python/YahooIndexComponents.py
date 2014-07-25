from lxml.html import parse
from urllib.request import urlopen
from pandas.io.parsers import TextParser

class YahooIndexComponents:
    
    def __init__(self, ticker):
        parsed = parse(urlopen('http://finance.yahoo.com/q/cp?s=%s' % ticker))
        doc = parsed.getroot()
        self.tables = doc.findall('.//table')
        self.table = self.tables[8]

    def _unpack(self, row, tag='td'):
        elts = row.findall('.//%s' % tag)
        return [val.text_content() for val in elts]

    def parse(self):
        rows = self.table.findall('.//tr')
        self.header = self._unpack(rows[0], tag='th')
        self.data = [self._unpack(r) for r in rows[1:]]
        return TextParser(self.data, names=self.header).get_chunk()

