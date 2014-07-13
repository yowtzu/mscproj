\l yahoo.q

getOHLCV:{[indexSym]
  syms:get hsym `$string[indexSym],"Constituents";
  res:yahoo[syms;2000.01.01;.z.d-1];
  res:update `p#sym, `g#date from `sym`date xasc res;
  hsym[indexSym] set res;   
  }

getOHLCV each (`$"FTSE";`$"GSPC");