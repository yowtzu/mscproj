import pandas as pd
import pandas.io.data as web
import datetime
from pandas.tools.plotting import autocorrelation_plot

pd.options.display.line_width=300
start = datetime.datetime(2006,1,1)
end = datetime.datetime(2013,12,31)

comps = YahooIndexComponents('^DJI').parse()
print(comps)

res = [web.DataReader(comp, 'yahoo', start,end) for comp in comps.Symbol]

res = pd.concat([res[x]["Adj Close"] for x in range(0,29)], axis=1)

from pandas.tools.plotting import autocorrelation_plot

plot(autocorrelation_plot(res.pct_change()))

#opinions = [YahooAnalystOpinionsFeed(comp).parse() for comp in comps.Symbol]
# keys=pd.concat(opinions).To.unique()
# values=

