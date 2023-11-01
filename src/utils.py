import html
import pandas as pd
import numpy as np
from sqlalchemy import text, create_engine
from transformers import pipeline
from transformers import AutoTokenizer
from tqdm import tqdm

def extraer_relato(lista_ndds, sql_connection):
    """
    Devuelve un dataframe que contiene la NDD y el relato de los hechos
    @lista_ndds: lista con las ndds a ser consultadas
    @sql_connection: objeto que crea la conexion con la base
    return dataframe con NDD y Relato
    """
    sql_query = text("""
                 SELECT den.codfisc AS 'NDD', bdd_enlace_externo.fnStripTags(den.obserinc) AS 'RELATO'
                 FROM fgn.denuncia_fiscalia AS den
                 WHERE den.estado=1 AND den.anulada='NO' AND den.codfisc IN :ndds_list
                 GROUP BY den.codfisc;
                 """)
    sql_query = sql_query.bindparams(ndds_list=tuple(lista_ndds))
    relatos = pd.read_sql(sql_query, sql_connection)
    relatos.RELATO = relatos.RELATO.str.lower()
    relatos.RELATO = relatos.RELATO.apply(lambda x: html.unescape(x))
    # Removiendo xa0 que proviende de encodificacion Latin1 ISO8859-1
    relatos.RELATO = relatos.RELATO.str.replace(u'\xa0', u' ')
    return relatos
    

def conectar_sql(big_data_bbdd=True):
    # F0s!Hu63
    if big_data_bbdd:
        engine_maria_db = create_engine("mysql+pymysql://falconiel:BebuSuKO@192.168.152.197", pool_recycle=3600)
        print("conectando con big data database....")
    else:
        engine_maria_db = create_engine("mysql+pymysql://falconiel:F0s!Hu63@192.168.152.8", pool_recycle=3600)  # N27a34v1
        print("conectando con proxy database....")
    print(engine_maria_db.connect())
    return engine_maria_db
    
    
def format_crimestory(relato_label, dataf):
    
    dataf[relato_label] = dataf[relato_label].str.lower()
    dataf[relato_label] = dataf[relato_label].str.replace("[^A-Za-z0-9áéíóúñ]+", " ", regex=True)
    dataf[relato_label] = dataf[relato_label].str.strip()
    
    
def load_text_classification_model(path2model, seq_len, threshold_words_qty):
    """_load_text_classification_model: loads the machine learning model for text classification

    Args:
        path2model (_type_): location in disk of machine learning model:/home/falconiel/ML_Models/robbery_tf20221113 
        seq_len (_type_): maximum sequence of the text for distilbert model
        threshold_words_qty (_type_): if the text has less words than this value will be ignored

    Returns:
        _type_: huggingface classification pipe
    """
    model_ckpt = "distilbert-base-multilingual-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt,  padding="max_length", truncation=True, max_length=seq_len)
    return pipeline("text-classification", model=path2model, tokenizer=tokenizer)
    
    
def predictLabelAndScore(relato, classifier):
    y_hat_dict = classifier(relato, truncation=True)[0]
    label = y_hat_dict['label']
    score = y_hat_dict['score']
    return label, score
    

def predictLabelAndScoreDaaS(relato, classifier, status, actual_label, actual_score, words_qty, threshold_words_qty):
    if status == 0:
        # when status is 0 make a prediction if there are enough words
        if words_qty >= threshold_words_qty:
            y_hat_dict = classifier(relato, truncation=True)[0]
            label = y_hat_dict['label']
            score = y_hat_dict['score']
            status = 1
        else:
            label = "N/A"
            score = 0
            status = status
    else:
        # when status is different from 0, nothing changes
        label = actual_label
        score = actual_score
        status = status
    return label, score, status
    
    
def words_qty(dataf, relato_label):
    """_words_qty_: returns the number of words of a text column

    Args:
        dataf (_type_): dataframe
        relato_label (_type_): the column where text to be classified is written
    """
    dataf["d_CANTIDAD_PALABRAS"] = dataf[relato_label].apply(lambda x: len(x.split(' ')))
    

def predict_text_class(dataf, model, label_relato= 'd_RELATO_SIAF',label_name='d_DELITOS_SEGUIMIENTO', words_qty_label='d_CANTIDAD_PALABRAS', threshold_words_qty=50):
    """predict_text_class
        Classfies the given text according to model
    Args:
        dataf (_type_): dataframe that contains the text to be classified
        model (_type_): model to be used. must be a hugging face pipeline pre-loaded
        label_relato (str, optional): name of the column with the text to be classified. Defaults to 'd_RELATO_SIAF'.
        label_name (str, optional): name of the new column to be returned. Defaults to 'd_DELITOS_SEGUIMIENTO'.
        words_qty_label (str, optional): name of the column with the number of words. Defaults to 'd_CANTIDAD_PALABRAS'.
        threshold_words_qty (int, optional): constant value to consider valid text to be classified. Defaults to 50.
    """
    dataf[[label_name, label_name+'_SCORE']] = dataf.apply(lambda x: predictLabelAndScore(relato=x[label_relato], classifier=model) if x[words_qty_label] >=threshold_words_qty else ("N/A", 0), axis=1, result_type='expand')
    
    
def predict_text_class_tqdm(dataf, model, label_relato= 'd_RELATO_SIAF',label_name='d_DELITOS_SEGUIMIENTO', words_qty_label='d_CANTIDAD_PALABRAS', threshold_words_qty=50):
    """predict_text_class_tqdm
        Classfies the given text according to model. It includes tqdm to evaluate progress
    Args:
        dataf (_type_): dataframe that contains the text to be classified
        model (_type_): model to be used. must be a hugging face pipeline pre-loaded
        label_relato (str, optional): name of the column with the text to be classified. Defaults to 'd_RELATO_SIAF'.
        label_name (str, optional): name of the new column to be returned. Defaults to 'd_DELITOS_SEGUIMIENTO'.
        words_qty_label (str, optional): name of the column with the number of words. Defaults to 'd_CANTIDAD_PALABRAS'.
        threshold_words_qty (int, optional): constant value to consider valid text to be classified. Defaults to 50.
    """
    tqdm.pandas()
    dataf[[label_name, label_name+'_SCORE']] = dataf.progress_apply(lambda x: predictLabelAndScore(relato=x[label_relato], classifier=model) if x[words_qty_label] >=threshold_words_qty else ("N/A", 0), axis=1, result_type='expand')
    
    
# def update_predictLabelAndScoreDaaS(label_relato, classifier, words_qty_label,threshold_words_qty, y_predicted, score, status):
#     if status == 0:
#         if words_qty_label >= threshold_words_qty:
#             return predictLabelAndScoreDaaS(relato=label_relato, classifier=classifier)
#         else:
#             return 'N/A', 0, status
#     else:
#         return y_predicted, score, status


def predict_text_class_DaaS_tqdm(dataf, model, label_relato= 'RELATO',label_name='delitos_seguimiento_predicted', label_score='delitos_seguimiento_predicted_SCORE', words_qty_label='d_CANTIDAD_PALABRAS', threshold_words_qty=50, status='ESTADO_ML'):
    """predict_text_class_tqdm
        Classfies the given text according to model. It includes tqdm to evaluate progress
    Args:
        dataf (_type_): dataframe that contains the text to be classified
        model (_type_): model to be used. must be a hugging face pipeline pre-loaded
        label_relato (str, optional): name of the column with the text to be classified. Defaults to 'd_RELATO_SIAF'.
        label_name (str, optional): name of the new column to be returned. Defaults to 'd_DELITOS_SEGUIMIENTO'.
        words_qty_label (str, optional): name of the column with the number of words. Defaults to 'd_CANTIDAD_PALABRAS'.
        threshold_words_qty (int, optional): constant value to consider valid text to be classified. Defaults to 50.
        status str: name of the column that stores the state of the ndd. If 0, the ndd has not been predicted or is a new case, if 1, the ndd has been predicted previously and can be skipped
    """
    tqdm.pandas()
    # HAY UN PROBLEMA EN EL APPLY, SE REQUIERE QUE ME PONGA N/A SI ERA CANTIDAD PALABRAS MENOS DE 50 Y ESTADO 0, PERO SI EL ESTADO ML ES 1, DEBE QUEDAR EL MISMO VALOR
    # dataf[[label_name, label_name+'_SCORE', status]] = dataf.progress_apply(lambda x: predictLabelAndScoreDaaS(relato=x[label_relato], classifier=model) if x[words_qty_label] >=threshold_words_qty and x[status]==0 else ("N/A", 0, 0), axis=1, result_type='expand')
    dataf[[label_name, label_name+'_SCORE', status]] = dataf.progress_apply(lambda x: predictLabelAndScoreDaaS(relato=x[label_relato], classifier=model, actual_label=x[label_name], actual_score=x[label_score], words_qty=x[words_qty_label], threshold_words_qty=threshold_words_qty, status=x[status]), axis=1, result_type='expand')


def asamblea4_ndd_chunked(sql_conn, lista_ndds, chunk_size=1000):
    """obtiene el detalle de la ndd de acuerdo al script sql de asamblea 4
    pero separa la lista de ndds en chunks de chunk_size 1000 y realiza la lectura
    de cada una de estas partes para luego unir en una sola respuesta

    Args:
        sql_conn (int, optional): conexión a la base de datos
        lista_ndds (int, optional): lista con las ndds a consultar
        chunk_size (int): tamaño del split de la lista de ndds

    Return:
        dataframe con el resulado en siaf de las ndds solicitadoas
    """
    sql_query = text("""SELECT 
                df.codfisc AS NDD, 
                df.fecha AS Fecha_Registro,
                df.hora AS Hora_Registro,
                df.fechainc AS Fecha_Incidente,
                df.horainc AS Hora_Incidente,
                If(df.tentativa='N','No','Si') AS Tentativa,
                df.direccion AS Direccion,
                del.gen_delito_tipopenal AS Presunto_Delito,
                del.gen_delito_concatenado AS 'Presunto Delito (Circunstancia Modificatoria)',
                (SELECT fisc.fxz_descripcion From fgn.fiscalias as fisc where df.fiscalias = fisc.fxz_codigo) as 'Fiscalia',
                fzxu.descripcion AS Fiscalia_Especializada,
                (SELECT fiscalia.gen_canton.can_descripcion From fiscalia.gen_canton Where fiscalia.gen_canton.can_codigo = df.ciudades ) as 'Ciudad',
                fzxu.pro_descripcion AS PROVINCIA,
                (SELECT fiscalia.gen_canton.can_descripcion From fiscalia.gen_canton   Where fiscalia.gen_canton.can_iso = substr(df.parroquias,1,4)    and fiscalia.gen_canton.can_estado = '1' ) as 'Canton',
                (SELECT par_nombre From fiscalia.gen_parroquia as parr where df.parroquias = parr.par_codigo) as 'Parroquia',
                (SELECT ti.descripcion From fgn.tipoincidente as ti where df.tipoincidente=ti.cdg) as 'Tipo',
                if (df.tipodelito=0, 'No Flagrante','Flagrante' ) AS NyF,
                CONCAT ('Fiscalia ',fzxu.fzxu_num_fiscalia) AS numero_fiscalia,
                fzxu.fxz_descripcion AS edificio,
                fzxu.fzxu_nombre_fiscal as nombre_fiscal,
                (SELECT tp2_descripcion From fiscalia.gen_tipos2 as fue where df.fuero = fue.tp2_codigo) as 'Fuero',
                CASE fso.fso_etapas when 265 then concat('INVESTIGACION PREVIA') when 266 then concat('INSTRUCCION FISCAL') when 267 then concat('PREPARATORIA DE JUICIO') when 268 then concat('JUICIO') when 1065 then concat('IMPUGNACION') else 'SIN ACCIONES' END as Etapa_procesal, 
                CASE WHEN tip2.tp2_descripcion IS NULL THEN 'POR APERTURAR INVETIGACION PREVIA' ELSE tip2.tp2_descripcion END AS Estado_Procesal,
                DATE_FORMAT(fso.fso_fecha_estado_procesal, '%Y-%m-%d') Fecha_estado_procesal,
                (SELECT count(distinct exndd.codigo_exn) as numero_diligencias from milenium.proceso_ndd as pn INNER JOIN milenium.experticiaxndd exndd ON exndd.cod_uni_exp=pn.codigo_proc where pn.estado = 1 and exndd.estado = 1 AND exndd.NDD=df.codfisc) AS 'IMPULSOS_DILIGENCIAS',
                (SELECT count(pn.ndd) as impulsos from milenium.proceso_ndd as pn where pn.estado = 1 and pn.ndd=df.codfisc) AS 'IMPULSOS',
                ult.nua_ult_accion AS Ultima_accion,
                ult.nua_fecha_ult_accion AS Fecha_Ultima_Accion,
                (SELECT COUNT(inv1.id_involucrado) FROM fgn.involucrado inv1 WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (34,193)) AS APREHENDIDO,
                (SELECT COUNT(inv1.id_involucrado) FROM fgn.involucrado inv1 WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (37,35)) AS SOSPECHOSO,
                (SELECT COUNT(inv1.id_involucrado) FROM fgn.involucrado inv1 WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (53,194)) AS PROCESADO,
                (SELECT COUNT(inv1.id_involucrado) FROM fgn.involucrado inv1 WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (995)) AS DETENIDO,
                (SELECT COUNT(inv1.id_involucrado) FROM fgn.involucrado inv1 WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (49)) AS IMPUTADO,
                df.numeroinf,
                (SELECT tipinc.descripcion from fgn.tipoincidente tipinc WHERE df.tipoincidente=tipinc.cdg LIMIT 1) AS Tipo_incidente,
                (SELECT MAX(ip.ind_fecha_ini) FROM fgn.fa2_indagacion_previa ip WHERE ip.ndd_codigo=fso.ndd_codigo  AND ip.ind_estado=1 AND ip.cod_fzxu=fso.fzxu_codigo) Fecha_IP_Inicio,
                (SELECT MAX(ip.ind_fecha_fin) FROM fgn.fa2_indagacion_previa ip WHERE ip.ndd_codigo=fso.ndd_codigo  AND ip.ind_estado=1 AND ip.cod_fzxu=fso.fzxu_codigo) Fecha_IP_Fin,
                (SELECT MAX(ins.ins_fecha_inicio) FROM fgn.fa2_instruccion ins WHERE ins.ndd_codigo=fso.ndd_codigo  AND ins.ins_estado=1 AND ins.cod_fzxu=fso.fzxu_codigo) Fecha_INS_Inicio,
                (SELECT MAX(ins.ins_fecha_cierre) FROM fgn.fa2_instruccion ins WHERE ins.ndd_codigo=fso.ndd_codigo  AND ins.ins_estado=1 AND ins.cod_fzxu=fso.fzxu_codigo) Fecha_INS_Fin,
                (SELECT MAX(ins.ins_tiempo) FROM fgn.fa2_instruccion ins WHERE ins.ndd_codigo=fso.ndd_codigo  AND ins.ins_estado=1 AND ins.cod_fzxu=fso.fzxu_codigo) Tiempo_INS,
                bdd_enlace_externo.quitaEntities(bdd_enlace_externo.fnStripTags(df.obserinc)) AS Relato,
                (SELECT max(pro.pro_descripcion) FROM fiscalia.gen_provincia pro WHERE pro.pro_codigo=df.provincias) AS PROVINCIA_INCIDENTE,
                (SELECT max(can.can_descripcion) FROM fiscalia.gen_canton can WHERE can.can_iso=df.cantones) AS CANTON_INCIDENTE
                FROM fgn.denuncia_fiscalia df
                INNER JOIN fgn.fiscal_sorteado fso ON fso.ndd_codigo=df.codfisc
                INNER JOIN fgn.gen_fiscalia_zonal_unidad_vista fzxu ON fzxu.fzxu_codigo=fso.fzxu_codigo
                INNER JOIN fgn.gen_delitos del ON del.gen_delito_secuencia=df.infraccion
                LEFT JOIN fgn.fa2_ndd_ult_accion ult ON ult.nua_ndd=df.codfisc  
                LEFT JOIN fiscalia.gen_tipos2 tip2 ON tip2.tp2_codigo=fso.fso_estado_procesal
                WHERE df.estado=1
                AND df.anulada='NO'
                AND fso.fso_estado<3
                -- AND df.fecha BETWEEN '2019-11-01' AND '2019-11-20'
                -- AND df.fechainc BETWEEN '2019-10-02' AND '2019-10-13'
                -- AND df.codfisc IN (
                and df.codfisc IN :ndds_list
                -- AND df.tipoincidente NOT IN (10,33)
                GROUP BY df.codfisc;""")
    lista_ndds_chunks = [lista_ndds[i:i+chunk_size] for i in range(0, len(lista_ndds), chunk_size)]
    print(f"Separando la lista {len(lista_ndds)} en {len(lista_ndds_chunks)} chunks")
    dataf_chunks = []
    for lista_ndd_chunk in tqdm(lista_ndds_chunks):
        sql_query = sql_query.bindparams(ndds_list=tuple(lista_ndd_chunk))
        tmp_df = pd.read_sql_query(sql_query, sql_conn)
        dataf_chunks.append(tmp_df)
    resp = pd.concat(dataf_chunks)
    resp.reset_index(inplace=True)
    return resp
    

def asamblea4_ndd_acumuladas_chunked(sql_conn, lista_ndds, chunk_size=1000):
    """obtiene el detalle de la ndd de acuerdo al script sql de asamblea 4
    pero separa la lista de ndds en chunks de chunk_size 1000 y realiza la lectura
    de cada una de estas partes para luego unir en una sola respuesta

    Args:
        sql_conn (int, optional): conexión a la base de datos
        lista_ndds (int, optional): lista con las ndds a consultar
        chunk_size (int): tamaño del split de la lista de ndds

    Return:
        dataframe con el resulado en siaf de las ndds solicitadoas
    """
    sql_query = text("""SELECT 
                        df.codfisc AS NDD, 
                        df.fecha AS Fecha_Registro,
                        df.hora AS Hora_Registro,
                        df.fechainc AS Fecha_Incidente,
                        df.horainc AS Hora_Incidente,
                        If(df.tentativa='N','No','Si') AS Tentativa,
                        df.direccion AS Direccion,
                        del.gen_delito_tipopenal AS Presunto_Delito,
                        del.gen_delito_concatenado AS 'Presunto Delito (Circunstancia Modificatoria)',
                        (SELECT fisc.fxz_descripcion From fgn.fiscalias as fisc where df.fiscalias = fisc.fxz_codigo) as 'Fiscalia',
                        fzxu.descripcion AS Fiscalia_Especializada,
                        (SELECT fiscalia.gen_canton.can_descripcion From fiscalia.gen_canton Where fiscalia.gen_canton.can_codigo = df.ciudades ) as 'Ciudad',
                        fzxu.pro_descripcion AS PROVINCIA,
                        (SELECT fiscalia.gen_canton.can_descripcion From fiscalia.gen_canton   Where fiscalia.gen_canton.can_iso = substr(df.parroquias,1,4)    and fiscalia.gen_canton.can_estado = '1' ) as 'Canton',
                        (SELECT par_nombre From fiscalia.gen_parroquia as parr where df.parroquias = parr.par_codigo) as 'Parroquia',
                        (SELECT ti.descripcion From fgn.tipoincidente as ti where df.tipoincidente=ti.cdg) as 'Tipo',
                        if (df.tipodelito=0, 'No Flagrante','Flagrante' ) AS NyF,
                        CONCAT ('Fiscalia ',fzxu.fzxu_num_fiscalia) AS numero_fiscalia,
                        fzxu.fxz_descripcion AS edificio,
                        fzxu.fzxu_nombre_fiscal as nombre_fiscal,
                        (SELECT tp2_descripcion From fiscalia.gen_tipos2 as fue where df.fuero = fue.tp2_codigo) as 'Fuero',
                        CASE fso.fso_etapas when 265 then concat('INVESTIGACION PREVIA') when 266 then concat('INSTRUCCION FISCAL') when 267 then concat('PREPARATORIA DE JUICIO') when 268 then concat('JUICIO') when 1065 then concat('IMPUGNACION') else 'SIN ACCIONES' END as Etapa_procesal, 
                        CASE WHEN tip2.tp2_descripcion IS NULL THEN 'POR APERTURAR INVETIGACION PREVIA' ELSE tip2.tp2_descripcion END AS Estado_Procesal,
                        (SELECT count(distinct exndd.codigo_exn) as numero_diligencias from milenium.proceso_ndd as pn INNER JOIN milenium.experticiaxndd exndd ON exndd.cod_uni_exp=pn.codigo_proc where pn.estado = 1 and exndd.estado = 1 AND exndd.NDD=df.codfisc) AS 'IMPULSOS_DILIGENCIAS',
                        ult.nua_ult_accion AS Ultima_accion,
                        ult.nua_fecha_ult_accion AS Fecha_Ultima_Accion,
                        (SELECT COUNT(inv1.id_involucrado) FROM fgn.involucrado inv1 WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (34,193)) AS APREHENDIDO,
                        (SELECT COUNT(inv1.id_involucrado) FROM fgn.involucrado inv1 WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (37,35)) AS SOSPECHOSO,
                        (SELECT COUNT(inv1.id_involucrado) FROM fgn.involucrado inv1 WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (53,194)) AS PROCESADO,
                        (SELECT COUNT(inv1.id_involucrado) FROM fgn.involucrado inv1 WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (995)) AS DETENIDO,
                        (SELECT COUNT(inv1.id_involucrado) FROM fgn.involucrado inv1 WHERE inv1.codfisc=df.codfisc AND inv1.inv_estado=1 AND inv1.inv_idtipo IN (49)) AS IMPUTADO,
                        df.numeroinf,
                        (SELECT tipinc.descripcion from fgn.tipoincidente tipinc WHERE df.tipoincidente=tipinc.cdg LIMIT 1) AS Tipo_incidente,
                        bdd_enlace_externo.quitaEntities(bdd_enlace_externo.fnStripTags(df.obserinc)) AS Relato,
                        fso.fso_acumulado
                        FROM fgn.denuncia_fiscalia df
                        INNER JOIN fgn.fiscal_sorteado fso ON fso.ndd_codigo=df.codfisc
                        INNER JOIN fgn.gen_fiscalia_zonal_unidad_vista fzxu ON fzxu.fzxu_codigo=fso.fzxu_codigo
                        INNER JOIN fgn.gen_delitos del ON del.gen_delito_secuencia=df.infraccion
                        LEFT JOIN fgn.fa2_ndd_ult_accion ult ON ult.nua_ndd=df.codfisc  
                        LEFT JOIN fiscalia.gen_tipos2 tip2 ON tip2.tp2_codigo=fso.fso_estado_procesal
                        WHERE df.estado=1
                        AND df.anulada='NO'
                        AND fso.fso_estado=6
                        -- AND df.fecha BETWEEN '2019-11-01' AND '2019-11-20'
                        -- AND df.fechainc BETWEEN '2019-10-02' AND '2019-10-13'
                        -- AND df.codfisc IN (
                        AND df.codfisc IN :ndds_list
                        -- AND df.tipoincidente NOT IN (10,33)
                        GROUP BY df.codfisc;""")
    lista_ndds_chunks = [lista_ndds[i:i+chunk_size] for i in range(0, len(lista_ndds), chunk_size)]
    print(f"Separando la lista {len(lista_ndds)} en {len(lista_ndds_chunks)} chunks")
    dataf_chunks = []
    for lista_ndd_chunk in tqdm(lista_ndds_chunks):
        sql_query = sql_query.bindparams(ndds_list=tuple(lista_ndd_chunk))
        tmp_df = pd.read_sql_query(sql_query, sql_conn)
        dataf_chunks.append(tmp_df)
    resp = pd.concat(dataf_chunks)
    resp.reset_index(inplace=True)
    return resp
    
    
def save_df_in_sql(name_table, dataf, index=False, where="bigData", database="reportes"):
    """
    This function saves a dataframe as a Table in a MariaDB MySQL database
    where: server that has the sql databases
    """    
    if where=="bigData":
        engine_maria_db = create_engine(f"mysql+pymysql://falconiel:BebuSuKO@192.168.152.197:3306/{database}"+"?charset=utf8mb4")
    else:
        engine_maria_db = create_engine(f"mysql+pymysql://falconiel:F0s!Hu63@192.168.152.8:3306/{database}"+"?charset=utf8mb4")
    print("conexion con base es: {}".format(engine_maria_db.connect()))
    dataf.to_sql(name_table, engine_maria_db, if_exists='replace', index=index, chunksize=1000)
    with engine_maria_db.connect() as conn:
        conn.execute(f'ALTER TABLE `{database}`.`{name_table}` CONVERT TO CHARACTER SET utf8 COLLATE utf8_general_ci;')