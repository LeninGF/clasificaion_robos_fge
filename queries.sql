# total de registros
SELECT count(*) FROM reportes.relatosPoliciaSiaf_20230830 re;
select * from reportes.relatosPoliciaSiaf_20230830 re limit 1000;
# cuenta de delitos siaf
select re.Presunto_Delito, 
count(re.NDD) as 'Total',
CONCAT(ROUND(COUNT(re.NDD) / SUM(COUNT(re.NDD)) OVER() * 100, 2), '%') AS 'Percentage'
from reportes.relatosPoliciaSiaf_20230830 re 
group by re.Presunto_Delito
ORDER BY COUNT(re.NDD) DESC;

## cuenta de delitos comision estadistica
select re.Tipo_Delito_PJ_comision, 
count(re.NDD) as 'Total',
CONCAT(ROUND(COUNT(re.NDD) / SUM(COUNT(re.NDD)) OVER() * 100, 2), '%') AS 'Percentage' 
from reportes.relatosPoliciaSiaf_20230830 re 
group by re.Tipo_Delito_PJ_comision
ORDER BY COUNT(re.NDD) DESC;
