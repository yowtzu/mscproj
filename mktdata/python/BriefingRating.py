from lxml.html import parse
from urllib.request import urlopen
from pandas.io.parsers import TextParser

class YahooIndexComponents:
    
    def __init__(self, ticker):
        parsed = parse(urlopen('http://www.briefing.com/investor/calendars/upgrades-downgrades/ratings-systems/'))
        doc = parsed.getroot()
        self.tables = doc.findall('.//table')
        self.table = self.tables[0]

    def _unpack(self, row, tag='td'):
        elts = row.findall('.//%s' % tag)
        return [val.text_content() for val in elts]

    def parse(self):
        rows = self.table.findall('.//tr')
        d = dict(zip([self._unpack(r)[0] for r in rows[1::2]], [self._unpack(r)[1:] for r in rows[2::2]]))
        return d
