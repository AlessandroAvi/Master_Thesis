#include "main.h" 


int sample_A[600] = {845,851,856,881,910,899,901,909,914,902,894,888,880,829,813,795,787,768,726,707,678,
                     646,639,615,591,564,552,553,501,500,468,494,476,452,465,377,381,485,453,435,432,
                     497,519,519,532,549,545,565,587,597,600,602,623,633,616,626,627,630,635,658,677,
                     682,681,685,682,682,700,715,699,717,702,695,704,713,704,704,691,672,670,695,717,
                     707,676,631,625,631,623,622,632,636,617,611,650,630,640,645,643,650,635,609,617,
                     629,639,641,627,620,614,596,605,600,592,592,591,587,581,577,584,593,601,610,609,
                     615,616,617,623,632,631,637,650,654,647,655,658,665,678,687,704,714,717,714,709,
                     704,706,711,714,719,707,702,705,701,700,698,708,714,727,737,748,749,748,759,768,
                     766,733,727,730,739,732,725,731,718,696,688,673,680,669,658,660,647,654,646,639,
                     643,641,644,653,646,633,632,626,623,635,622,603,595,599,604,598,591,579,583,-637,
                     -631,-624,-619,-616,-608,-624,-641,-635,-678,-689,-694,-730,-848,-863,-834,-790,-778,-761,-739,-754,
                     -738,-709,-661,-642,-623,-620,-597,-607,-521,-559,-620,-503,-445,-425,-407,-403,-402,-397,-436,-436,
                     -437,-498,-548,-568,-562,-556,-533,-512,-507,-496,-475,-483,-483,-494,-480,-468,-490,-521,-529,-572,
                     -590,-573,-557,-566,-588,-585,-603,-624,-600,-570,-539,-523,-493,-558,-577,-557,-550,-568,-515,-493,
                     -523,-568,-714,-756,-769,-760,-773,-747,-704,-668,-567,-494,-352,-318,-322,-398,-521,-565,-598,-625,
                     -653,-647,-618,-595,-564,-549,-551,-538,-534,-563,-598,-596,-616,-644,-676,-677,-665,-649,-627,-607,
                     -566,-559,-586,-618,-638,-647,-649,-629,-605,-600,-597,-593,-585,-574,-541,-508,-481,-475,-415,-400,
                     -392,-384,-391,-392,-379,-392,-417,-422,-473,-534,-588,-604,-615,-612,-575,-548,-543,-563,-564,-568,
                     -553,-556,-544,-531,-547,-559,-532,-534,-565,-591,-617,-612,-618,-665,-703,-693,-716,-719,-696,-647,
                     -590,-595,-604,-589,-573,-536,-556,-562,-552,-502,-476,-504,-545,-545,-436,-415,-442,-466,-455,348,
                     355,352,348,430,476,489,477,459,446,441,428,414,390,365,320,274,233,196,185,164,
                     120,74,30,33,33,34,109,-23,164,103,-40,-38,3,45,57,259,222,294,365,380,
                     289,247,220,222,255,323,318,301,289,283,255,246,258,276,292,288,282,272,269,269,
                     283,291,308,340,406,425,451,468,533,542,519,484,446,431,505,619,714,734,529,449,
                     451,491,634,712,785,828,835,754,722,714,662,585,608,631,656,694,822,908,941,926,
                     874,776,786,820,860,884,807,772,772,792,792,809,851,876,874,838,809,791,803,835,
                     886,915,928,922,888,872,845,831,861,962,995,1001,978,913,869,834,815,812,836,844,
                     832,799,769,707,698,685,668,643,640,648,651,659,677,681,685,697,692,596,537,514,
                     529,593,594,589,569,529,449,444,454,453,436,390,356,350,342,348,342,335,341,340,
                     284,240,212,225,267,292,250,196,141,148,192,203,176,131,93,101,119,131,142};


int sample_E[600] = {1174,1128,1094,1072,1062,1055,1006,971,953,931,890,821,801,788,767,712,701,711,695,666,653,
                     647,619,589,568,524,512,487,481,483,440,404,403,383,338,332,319,309,299,322,337,
                     333,335,394,412,415,443,474,528,539,557,603,652,669,675,695,738,759,805,854,859,
                     867,916,951,1007,1008,962,904,864,825,777,706,659,627,583,535,500,466,436,437,436,
                     462,494,526,570,723,823,921,1014,1129,1361,1462,1551,1622,1709,1716,1698,1657,1604,1466,1394,
                     1311,1224,1146,1000,927,857,790,657,578,498,436,390,303,249,226,196,146,95,77,52,
                     33,24,8,0,-2,8,23,5,23,57,94,139,209,300,376,540,638,757,900,1008,
                     1289,1431,1548,1649,1727,1788,1752,1686,1614,1428,1342,1264,1181,1089,973,918,858,783,722,577,
                     518,463,416,346,290,247,212,186,125,105,96,74,60,60,68,83,100,136,153,162,
                     180,200,263,291,338,385,423,540,605,658,727,881,942,996,1024,1049,1113,1130,1163,-503,
                     -508,-550,-559,-543,-534,-526,-509,-489,-491,-509,-498,-455,-426,-442,-490,-468,-421,-422,-474,-564,
                     -597,-623,-616,-578,-458,-411,-368,-337,-302,-230,-212,-188,-124,-74,-66,-74,-98,-132,-206,-191,
                     -173,-163,-198,-207,-226,-263,-293,-319,-369,-432,-464,-413,-416,-419,-377,-273,-258,-305,-384,-493,
                     -570,-630,-774,-904,-947,-897,-925,-1059,-1218,-1375,-1520,-1548,-1570,-1563,-1515,-1303,-1155,-988,-892,-628,
                     -445,-293,-155,-44,120,228,314,368,433,625,716,785,846,952,973,953,903,867,656,546,
                     436,334,263,114,22,-75,-162,-257,-321,-459,-570,-630,-662,-657,-696,-784,-838,-917,-993,-1095,
                     -1167,-1319,-1423,-1526,-1579,-1565,-1465,-1362,-1254,-1162,-1095,-940,-891,-814,-710,-494,-321,-72,201,440,
                     873,1137,1372,1566,1687,1851,1895,1859,1691,1158,940,754,574,369,5,-148,-267,-407,-562,-805,
                     -902,-966,-1033,-1170,-1170,-1188,-1237,-1257,-1297,-1292,-1297,-1334,-1381,-1400,-1440,-1473,-1463,-1374,-1335,-1291,
                     -1235,-1167,-1036,-932,-791,-648,-514,-234,-94,88,278,486,535,616,689,748,882,864,822,278,
                     156,113,66,46,77,297,370,378,350,299,141,75,49,47,51,59,107,191,273,290,
                     256,216,173,123,57,39,32,67,339,483,593,660,678,589,496,417,360,323,353,417,
                     463,448,344,263,172,112,88,240,346,367,294,197,209,191,127,57,114,139,205,364,
                     558,794,773,732,827,1098,1194,1263,1313,1351,1421,1413,1357,1283,1143,960,965,992,902,732,
                     638,614,627,640,652,670,721,761,771,790,855,942,1004,961,875,778,685,595,468,410,
                     366,339,305,241,220,201,207,349,391,357,259,155,92,116,169,235,278,309,324,343,
                     388,637,807,950,1037,1069,1023,929,788,674,637,612,563,541,560,623,648,690,794,928,
                     1120,1180,1202,1166,1045,645,418,210,30,-61,-13,54,125,191,315,371,431,493,511,458,
                     402,321,227,106,69,53,53,76,206,268,329,377,408,478,532,585,630,665,653,618,
                     564,509,464,481,508,548,596,633,689,752,800,846,890,924,905,812,516,396,347};


int sample_I[600] = {398,397,392,401,408,402,391,390,381,366,362,356,348,355,342,326,313,310,303,291,297,
                     300,312,313,307,313,356,370,383,398,415,434,441,442,440,424,425,434,424,403,424,
                     434,434,426,432,451,464,467,467,476,471,462,460,456,458,451,450,458,466,450,434,
                     422,421,414,412,402,383,366,349,357,353,347,359,354,348,358,373,372,385,395,387,
                     384,385,390,400,414,418,423,436,442,435,440,452,449,451,474,466,494,512,500,542,
                     526,523,540,552,562,558,566,577,630,640,646,653,669,723,768,834,846,868,964,1008,
                     1057,1117,1231,1257,1267,1248,1221,1170,1141,1123,1099,1062,978,947,917,887,833,796,762,730,
                     700,633,584,546,510,472,449,443,426,404,388,367,361,361,346,316,284,252,249,266,
                     214,211,226,216,175,166,168,172,157,123,115,109,104,117,142,147,165,187,223,246,
                     275,293,298,328,360,399,422,456,476,460,490,530,580,583,598,612,634,672,692,-577,
                     -581,-563,-547,-548,-571,-610,-652,-686,-751,-825,-869,-935,-994,-1026,-1064,-1122,-1135,-1124,-1102,-1106,
                     -1126,-1092,-1053,-1033,-1005,-968,-937,-912,-892,-900,-854,-821,-799,-770,-742,-753,-762,-760,-763,-747,
                     -723,-699,-683,-626,-622,-628,-657,-659,-613,-596,-573,-549,-515,-389,-382,-389,-394,-421,-450,-463,
                     -445,-408,-365,-366,-348,-343,-346,-331,-319,-316,-330,-341,-347,-383,-414,-428,-462,-498,-519,-525,
                     -533,-590,-625,-632,-628,-634,-625,-621,-603,-568,-567,-550,-540,-562,-598,-592,-568,-565,-576,-524,
                     -511,-519,-514,-514,-504,-475,-482,-498,-552,-586,-626,-652,-687,-729,-715,-755,-745,-720,-633,-548,
                     -424,-344,-273,-218,-195,-178,-184,-191,-181,-189,-180,-164,-158,-154,-134,-104,-16,15,40,57,
                     42,-69,-145,-214,-276,-340,-437,-459,-471,-503,-538,-543,-538,-559,-600,-593,-629,-691,-713,-698,
                     -747,-792,-832,-855,-872,-925,-981,-1001,-1016,-1078,-1091,-1083,-1069,-1044,-992,-994,-986,-963,-907,-914,
                     -924,-912,-886,-856,-847,-826,-788,-713,-669,-679,-674,-620,-574,-551,-506,-473,-429,-358,-361,701,
                     702,646,610,608,651,698,728,765,808,836,860,917,948,992,1029,1029,1014,995,1000,985,
                     920,924,949,961,939,907,922,936,930,850,799,752,713,694,675,658,683,749,776,647,
                     600,589,560,434,405,433,499,570,599,588,547,486,421,309,277,263,269,307,299,272,
                     226,186,124,110,104,115,129,95,106,140,182,216,197,155,130,149,207,208,222,248,
                     282,279,264,235,222,238,235,246,259,246,181,213,265,306,379,403,384,396,431,533,
                     592,584,504,429,499,582,636,643,720,841,977,1079,1117,1020,947,909,1006,1198,1619,1727,
                     1715,1598,1402,1404,1408,1448,1490,1393,1279,1176,1115,1083,1008,938,872,815,688,615,547,494,
                     470,545,608,637,652,624,475,429,408,407,432,422,381,344,348,471,541,534,465,422,
                     359,322,315,392,568,564,528,538,586,658,674,686,682,654,659,663,651,636,601,554,
                     536,574,645,733,732,712,701,727,729,744,749,733,722,721,710,690,666,625,607};


int sample_O[600] = {436,446,451,450,453,453,464,470,463,461,479,499,511,506,509,517,513,483,504,521,521,
                     498,470,451,435,420,395,345,330,330,342,281,265,268,252,215,221,204,183,184,172,
                     135,134,156,95,40,54,26,-24,-104,-144,-157,-170,-210,-257,-259,-260,-263,-257,-259,-243,
                     -228,-197,-140,-121,-127,-86,-50,-4,15,49,75,101,114,110,125,158,209,200,179,175,
                     175,175,169,169,188,200,182,201,212,218,228,239,247,249,266,273,262,283,335,376,
                     363,381,356,392,438,428,427,452,489,481,472,467,457,478,455,499,494,499,609,632,
                     680,704,722,754,791,796,810,803,784,781,766,746,701,680,655,638,624,561,496,465,
                     433,385,306,258,222,190,142,104,61,45,27,-58,-98,-147,-199,-240,-328,-374,-413,-446,
                     -494,-510,-509,-510,-509,-475,-453,-438,-423,-389,-330,-298,-265,-224,-133,-82,-24,37,94,195,
                     256,337,420,482,626,688,751,811,940,1008,1053,1085,1110,1145,1167,1157,1138,1120,1094,-897,
                     -903,-884,-852,-864,-864,-835,-787,-747,-728,-741,-739,-683,-652,-603,-554,-526,-435,-376,-310,-303,
                     -258,-267,-278,-292,-280,-290,-371,-393,-371,-302,-388,-476,-549,-626,-684,-655,-686,-752,-793,-846,
                     -918,-941,-984,-1031,-1141,-1199,-1278,-1377,-1640,-1719,-1825,-1955,-1998,-1998,-1998,-1998,-1998,-1998,-1998,-1998,
                     -1992,-1931,-1801,-1712,-1637,-1552,-1430,-1197,-1083,-1010,-944,-866,-631,-534,-447,-405,-350,-335,-263,-179,
                     -129,-6,62,129,127,121,164,194,203,216,245,200,156,122,130,24,-105,-201,-190,-131,
                     -172,-185,-273,-329,-344,-340,-397,-439,-469,-528,-616,-685,-697,-660,-519,-651,-849,-1045,-1354,-1440,
                     -1469,-1459,-1476,-1683,-1657,-1612,-1532,-1429,-1168,-1048,-963,-887,-625,-492,-405,-379,-363,-271,-255,-292,
                     -315,-296,-348,-422,-461,-431,-403,-394,-366,-374,-448,-608,-642,-647,-649,-706,-907,-965,-1044,-1168,
                     -1265,-1307,-1379,-1437,-1454,-1451,-1398,-1355,-1330,-1295,-1239,-1224,-1196,-1172,-1142,-1097,-1048,-1008,-979,-860,
                     -798,-757,-729,-691,-585,-499,-409,-315,-194,-152,-100,-39,32,132,113,95,83,67,-28,49,
                     40,45,72,82,86,87,114,116,97,90,111,209,225,203,171,160,122,60,20,46,
                     271,330,321,289,231,152,164,181,182,167,227,299,397,486,525,492,471,463,460,538,
                     614,679,623,669,736,745,846,935,1030,1011,960,932,960,1036,1049,1034,998,941,823,781,
                     772,794,845,870,926,887,820,711,665,655,669,687,646,548,422,361,502,605,594,465,
                     295,133,144,182,205,255,230,144,96,104,123,128,115,87,70,142,141,87,45,69,
                     241,258,219,202,369,472,534,557,597,714,741,691,623,382,576,731,896,1075,1581,1789,
                     1934,1998,1998,1939,1931,1975,1998,1998,1851,1715,1604,1503,1263,1134,1021,923,841,741,703,634,
                     564,490,370,298,201,90,-50,-26,6,49,103,216,231,199,141,101,137,196,267,328,
                     377,397,429,462,485,470,436,386,337,308,327,339,347,368,471,503,511,520,548,592,
                     597,622,674,736,825,840,838,804,703,693,703,703,673,516,466,452,457,450,372};


int sample_U[600] = {536,517,510,505,528,519,504,493,477,516,507,504,514,523,513,535,543,542,535,545,542,
                     546,547,553,556,552,554,553,548,536,540,538,537,523,531,533,527,523,522,521,519,
                     516,510,508,517,516,505,478,461,449,424,386,328,306,282,255,232,222,196,213,255,
                     188,197,200,193,177,153,137,140,153,128,92,102,101,87,90,73,65,76,67,48,
                     58,40,40,50,49,35,36,30,18,30,29,14,16,44,29,22,29,48,45,58,
                     77,72,96,111,119,121,126,153,177,196,201,217,225,225,216,220,240,251,244,235,
                     251,263,270,268,268,310,314,294,307,317,320,326,337,331,320,314,325,321,315,299,
                     290,288,291,284,284,297,279,279,268,265,273,289,305,291,297,306,319,324,350,368,
                     399,407,405,444,453,457,459,474,482,486,491,504,519,516,527,539,533,535,540,533,
                     532,535,535,536,536,530,530,527,522,520,522,517,520,523,519,512,507,510,513,-773,
                     -807,-798,-816,-811,-826,-808,-799,-806,-724,-873,-815,-816,-817,-811,-808,-829,-851,-838,-823,-820,
                     -811,-800,-798,-804,-808,-826,-840,-851,-846,-833,-819,-809,-801,-797,-785,-781,-779,-753,-742,-734,
                     -701,-665,-627,-596,-548,-482,-400,-351,-298,-266,-248,-277,-286,-297,-307,-307,-295,-334,-363,-404,
                     -548,-601,-638,-655,-697,-936,-1019,-1052,-1072,-1145,-1235,-1236,-1226,-1227,-1149,-1168,-1151,-1106,-1086,-1045,
                     -1076,-1137,-1136,-1124,-1082,-1053,-1056,-1090,-1226,-1199,-1141,-1139,-1115,-1087,-1076,-1069,-1046,-1029,-1047,-1039,
                     -1047,-1060,-1040,-1058,-1070,-1055,-1030,-941,-921,-926,-940,-928,-898,-890,-878,-867,-885,-891,-901,-905,
                     -918,-834,-812,-833,-872,-871,-879,-888,-892,-883,-901,-919,-919,-920,-913,-906,-931,-943,-918,-854,
                     -809,-769,-759,-745,-682,-676,-693,-718,-702,-636,-618,-585,-588,-595,-578,-566,-564,-576,-572,-564,
                     -546,-538,-561,-622,-620,-625,-656,-699,-709,-724,-738,-757,-786,-802,-806,-814,-835,-890,-898,-914,
                     -932,-905,-884,-879,-869,-860,-878,-866,-857,-852,-856,-865,-867,-862,-855,-848,-860,-869,-870,246,
                     381,428,403,371,390,401,366,301,155,62,88,113,141,166,212,216,244,266,264,251,
                     249,246,243,251,254,260,264,272,271,255,228,206,194,201,216,226,222,195,205,230,
                     253,261,223,222,249,275,265,222,182,157,140,174,222,275,301,270,202,176,135,149,
                     291,341,405,456,474,490,538,615,736,851,856,769,734,735,818,829,827,839,877,1011,
                     1048,1051,1006,949,923,913,905,907,967,1009,1033,1022,955,893,885,850,814,817,837,859,
                     900,942,893,839,796,778,780,758,763,811,876,932,893,835,757,675,598,622,661,679,
                     675,651,630,623,589,470,446,454,456,456,456,442,424,416,405,344,315,317,331,310,
                     266,213,187,193,186,191,204,213,202,122,77,38,49,115,107,63,13,-11,1,1,
                     13,21,15,-2,12,29,35,33,43,58,76,100,173,189,181,168,161,164,178,190,
                     195,176,152,126,112,113,154,168,165,143,117,116,138,159,167,139,121,113,119};