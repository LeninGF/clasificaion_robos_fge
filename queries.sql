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

## robosML
alter table DaaS.robosML add delitos_seguimiento_unified_origin TEXT,
add delitos_seguimiento_unified_siaf_origin TEXT,
add delitos_validados_unified_siaf_origin TEXT;

UPDATE DaaS.robosML t
INNER JOIN DaaS.robosML_predicted_tmp  s ON t.NDD = s.NDD
SET t.delitos_seguimiento_unified_origin = s.delitos_seguimiento_unified_origin,
t.delitos_seguimiento_unified_siaf_origin = s.delitos_seguimiento_unified_siaf_origin,
t.delitos_validados_unified_siaf_origin = s.delitos_validados_unified_siaf_origin;


UPDATE DaaS.robosML
SET delitos_seguimiento_unified_siaf = CASE
    WHEN delitos_seguimiento_unified_siaf = 'ROBO DE ACCESORIOS DE VEHICULOS' THEN 'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS'
    WHEN delitos_seguimiento_unified_siaf = 'ROBO DE BIENES A ENTIDAD PUBLICA' THEN 'OTROS ROBOS'
    WHEN delitos_seguimiento_unified_siaf =  'ROBO DE BIENES PERSONALES AL INTERIOR DEL VEHICULO' THEN 'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS'
    WHEN delitos_seguimiento_unified_siaf =  'ROBO DE VEHICULOS' THEN 'ROBO DE CARROS'
    WHEN delitos_seguimiento_unified_siaf =  'ROBO EN VIAS O CARRETERAS' THEN 'OTROS ROBOS'
    ELSE delitos_seguimiento_unified_siaf  -- Keep other values unchanged
END;

UPDATE DaaS.robosML
SET delitos_validados_unified_siaf = CASE
    WHEN delitos_validados_unified_siaf = 'ROBO DE ACCESORIOS DE VEHICULOS' THEN 'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS'
    WHEN delitos_validados_unified_siaf = 'ROBO DE BIENES A ENTIDAD PUBLICA' THEN 'ROBO EN INSTITUCIONES PUBLICAS'
    WHEN delitos_validados_unified_siaf =  'ROBO DE BIENES PERSONALES AL INTERIOR DEL VEHICULO' THEN 'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS'
    WHEN delitos_validados_unified_siaf =  'ROBO DE VEHICULOS' THEN 'ROBO DE CARROS'
    WHEN delitos_validados_unified_siaf =  'ROBO EN VIAS O CARRETERAS' THEN 'OTROS ROBOS'
    ELSE delitos_validados_unified_siaf  -- Keep other values unchanged
END;
