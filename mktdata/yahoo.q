yahoo:{[stocks;startDate;endDate] 
 / ensure that stocks is a list (it might be a single symbol) and then get rid of duplicates, if any
 stocks: distinct stocks,();  
 endDate: .z.d; / .z.d gives current date
 / parameter string for HTTP request
 0N!params: "&d=" , (string -1+`mm$endDate) / end month 
    , "&e=" , (string `dd$endDate) / end day 
    , "&f=" , (string `year$endDate) / end year
    , "&g=d&a=" , (string -1+`mm$startDate) / start month
    , "&b=" , (string `dd$startDate) / start day
    , "&c=" , (string `year$startDate) / start year
    , "&ignore=.csv";
 tbl:(); / initialize results table
 i:0;
 do[count stocks; /iterate over all the stocks
     stock: stocks[i];
     / send HTTP request for this stock; we get back a string
     txt: `:http://ichart.finance.yahoo.com "GET /table.csv?s=" , (string stock) , params , " http/1.0\r\nhost:ichart.finance.yahoo.com\r\n\r\n";
     pattern: "Date,Open"; / pattern to search for in the result string
     startindex: txt ss pattern; / the function ss finds the positions of a pattern in a string
    txt: startindex _ txt; / drop everything before the pattern (HHTP headers, etc)
    if[not count[txt];
    show "retry";
     / send HTTP request for this stock; we get back a string
     txt: `:http://ichart.finance.yahoo.com "GET /table.csv?s=" , (string stock) , params , " http/1.0\r\nhost:ichart.finance.yahoo.com\r\n\r\n";
     pattern: "Date,Open"; / pattern to search for in the result string
     startindex: txt ss pattern; / the function ss finds the positions of a pattern in a string
    txt: startindex _ txt; / drop everything before the pattern (HHTP headers, etc)
    ];
    txt:ssr[;"Adj Close";"AdjClose"] each txt;
    stocktable: ("DEEEEIE";enlist",")0:txt; / parse the string and create a table from it
    stocktable: update Sym:stock from stocktable; / add a column with name of stock
    tbl,: stocktable; / append the table for this stock to tbl
    i+:1
  ];
 tbl:`Date`Sym xasc select from tbl where not null Volume; / get rid of rows with nulls
 tbl:select date:Date, sym:Sym, open:Open, high:High, low:Low, close:Close, volume:Volume, adjOpen:Open*AdjClose%Close, adjHigh:High*AdjClose%Close, adjLow:Low*AdjClose%Close, adjClose:AdjClose from tbl } / order by date and stock
 