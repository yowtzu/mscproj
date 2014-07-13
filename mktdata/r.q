\l schema.q

upd:insert;

.u.x:(":5010";":5012")

.u.end:{t:tables `.;
    
  }

/ init schema and syc up from log file
.u.rep:{[x;y]
    .[;();:;]
  }

h:(hopen `$":",.u.x 0)
res: h "(.u.sub[`;`]; `.u `i`L)"
.u.rep . res