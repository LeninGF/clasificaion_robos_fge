import html
import pandas as pd
import numpy as np
from sqlalchemy import text, create_engine
from transformers import pipeline
from transformers import AutoTokenizer
from tqdm import tqdm


CATEGORIAS_DELITOS_SEGUIMIENTO = ['OTROS ROBOS',
                                    'ROBO A DOMICILIO',
                                    'ROBO A PERSONAS',
                                    'ROBO A UNIDADES ECONOMICAS', 
                                    'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS', 
                                    'ROBO DE CARROS', 
                                    'ROBO DE MOTOS']

CATEGORIAS_DELITOS_VALIDADOS = ['OTROS ROBOS', 
                                'ROBO A DOMICILIO', 
                                'ROBO A EMBARCACIONES DE ESPACIOS ACUATICOS', 
                                'ROBO A ESTABLECIMIENTOS DE COLECTIVOS U ORGANIZACIONES SOCIALES', 
                                'ROBO A INSTITUCIONES EDUCATIVAS', 
                                'ROBO A PERSONAS', 
                                'ROBO A UNIDADES ECONOMICAS', 
                                'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS', 
                                'ROBO DE CARROS', 
                                'ROBO DE MOTOS', 
                                'ROBO EN INSTITUCIONES PUBLICAS']

siaf_seguimiento_dict = {
    'OTROS ROBOS':'OTROS ROBOS',
    'ROBO A DOMICILIO':'ROBO A DOMICILIO',
    'ROBO A PERSONAS':'ROBO A PERSONAS',
    'ROBO A UNIDADES ECONOMICAS':'ROBO A UNIDADES ECONOMICAS',
    'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS':'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS',
    'ROBO DE CARROS':'ROBO DE CARROS',
    'ROBO DE MOTOS':'ROBO DE MOTOS',
    'NO_APLICA': 'REVIEW_LABEL',
    'ROBO DOMICILIOS': 'ROBO A DOMICILIO',
    'ROBO DE ACCESORIOS DE VEHÍCULOS': 'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS',
    'ROBO DE ACCESORIOS DE VEHICULOS': 'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS',
    'ROBO DE VEHÍCULOS': 'ROBO DE CARROS',
    'ROBO DE VEHICULOS': 'ROBO DE CARROS',
    'ROBO DE MOTOCICLETAS': 'ROBO DE MOTOS',
    'ROBO EN LOCALES COMERCIALES': 'ROBO A UNIDADES ECONOMICAS',
    'ROBO A BIENES DE UNIDADES ECONOMICAS': 'ROBO A UNIDADES ECONOMICAS',
    'ROBO A BIENES DE INSTITUCIONES EDUCATIVAS': 'OTROS ROBOS',
    'ROBO A BIENES DE INSTITUCION PUBLICA': 'OTROS ROBOS',
    'ROBO OTROS': 'OTROS ROBOS',
    'ROBO DE BIENES PERSONALES AL INTERIOR DEL VEHÍCULO': 'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS',
    'ROBO DE BIENES PERSONALES AL INTERIOR DEL VEHICULO': 'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS',
    'ROBO EN VÍAS O CARRETERAS': 'OTROS ROBOS',
    'ROBO EN VIAS O CARRETERAS': 'OTROS ROBOS',
    'ROBO DE BIENES A EMPRESA/FABRICA': 'ROBO A UNIDADES ECONOMICAS',
    'ROBO DE BIENES A ENTIDAD PÚBLICA': 'OTROS ROBOS',
    'ROBO DE BIENES A ENTIDAD PUBLICA': 'OTROS ROBOS',
    'ROBO DE BIENES A INSTITUCION EDUCATIVA': 'OTROS ROBOS',
    'ROBO A BIENES DE ESTABLECIMIENTOS DE COLECTIVOS U ORGANIZACIONES SOCIALES': 'OTROS ROBOS',
    'ROBO A EMBARCACIONES': 'OTROS ROBOS',
    'ROBO A BIENES DE INSTITUCIONES DE SALUD': 'OTROS ROBOS',
    'ROBO A BIENES DE ENTIDADES FINANCIERAS': 'ROBO A UNIDADES ECONOMICAS',
    'ROBO A VEHICULOS DE TRANSPORTE DE VALORES': 'OTROS ROBOS',
    'ROBO DE BIENES A ENTIDAD PRIVADA': 'ROBO A UNIDADES ECONOMICAS',
    'ROBO DE MOTORES EMBARCACIONES': 'OTROS ROBOS',
    'ROBO A BANCOS - ENTIDADES  FINANCIERAS': 'ROBO A UNIDADES ECONOMICAS',
    'HURTO A PERSONAS': 'REVIEW_LABEL',
    'ROBO A  BOTES PESQUEROS – YATES- FIBRAS-VELEROS ETC.': 'OTROS ROBOS',
    'HURTO A DOMICILIO': 'REVIEW_LABEL',
    'HURTO A MOTOS': 'REVIEW_LABEL',
    'HURTO A BIENES PUBLICOS': 'REVIEW_LABEL',
    'HURTO A CARROS': 'REVIEW_LABEL',
    'HURTO A ACCESORIOS': 'REVIEW_LABEL',
    'HURTO A BIENES DE UNIDADES ECONOMICAS': 'REVIEW_LABEL',
    'TRANSITO': 'REVIEW_LABEL',
    'ABIGEATO': 'REVIEW_LABEL',
    'HURTO A EMBARCACIONES O PARTES EN ESPACIOS ACUATICOS': 'REVIEW_LABEL',
    'HURTO A BIENES PATRIMONIALES': 'REVIEW_LABEL',
    'HURTO DE LO REQUISADO': 'REVIEW_LABEL',
    'ROBO A BIENES PATRIMONIALES': 'OTROS ROBOS',
    'HURTO DE BIENES DE USO POLICIAL O MILITAR': 'REVIEW_LABEL'}


siaf_validados_dict = {
    'OTROS ROBOS':'OTROS ROBOS',
    'ROBO A DOMICILIO':'ROBO A DOMICILIO',
    'ROBO A EMBARCACIONES DE ESPACIOS ACUATICOS':'ROBO A EMBARCACIONES DE ESPACIOS ACUATICOS',
    'ROBO A ESTABLECIMIENTOS DE COLECTIVOS U ORGANIZACIONES SOCIALES':'ROBO A ESTABLECIMIENTOS DE COLECTIVOS U ORGANIZACIONES SOCIALES',
    'ROBO A INSTITUCIONES EDUCATIVAS':'ROBO A INSTITUCIONES EDUCATIVAS',
    'ROBO A PERSONAS':'ROBO A PERSONAS',
    'ROBO A UNIDADES ECONOMICAS':'ROBO A UNIDADES ECONOMICAS',
    'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS':'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS',
    'ROBO DE CARROS':'ROBO DE CARROS',
    'ROBO DE MOTOS':'ROBO DE MOTOS',
    'ROBO EN INSTITUCIONES PUBLICAS':'ROBO EN INSTITUCIONES PUBLICAS',
    'NO_APLICA': 'REVIEW_LABEL',
    'ROBO DOMICILIOS': 'ROBO A DOMICILIO',
    'ROBO DE ACCESORIOS DE VEHÍCULOS': 'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS',
    'ROBO DE ACCESORIOS DE VEHICULOS': 'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS',
    'ROBO DE VEHÍCULOS': 'ROBO DE CARROS',
    'ROBO DE VEHICULOS': 'ROBO DE CARROS',
    'ROBO DE MOTOCICLETAS': 'ROBO DE MOTOS',
    'ROBO EN LOCALES COMERCIALES': 'ROBO A UNIDADES ECONOMICAS',
    'ROBO A BIENES DE UNIDADES ECONOMICAS': 'ROBO A UNIDADES ECONOMICAS',
    'ROBO A BIENES DE INSTITUCIONES EDUCATIVAS': 'ROBO A INSTITUCIONES EDUCATIVAS',
    'ROBO A BIENES DE INSTITUCION PUBLICA': 'ROBO EN INSTITUCIONES PUBLICAS',
    'ROBO OTROS': 'OTROS ROBOS',
    'ROBO DE BIENES PERSONALES AL INTERIOR DEL VEHÍCULO': 'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS',
    'ROBO DE BIENES PERSONALES AL INTERIOR DEL VEHICULO': 'ROBO DE BIENES, ACCESORIOS Y AUTOPARTES DE VEHICULOS',
    'ROBO EN VÍAS O CARRETERAS': 'OTROS ROBOS',
    'ROBO EN VIAS O CARRETERAS': 'OTROS ROBOS',
    'ROBO DE BIENES A EMPRESA/FABRICA': 'ROBO A UNIDADES ECONOMICAS',
    'ROBO DE BIENES A ENTIDAD PUBLICA': 'ROBO EN INSTITUCIONES PUBLICAS',
    'ROBO DE BIENES A ENTIDAD PÚBLICA': 'ROBO EN INSTITUCIONES PUBLICAS',
    'ROBO DE BIENES A INSTITUCION EDUCATIVA': 'ROBO A INSTITUCIONES EDUCATIVAS',
    'ROBO A BIENES DE ESTABLECIMIENTOS DE COLECTIVOS U ORGANIZACIONES SOCIALES': 'ROBO A ESTABLECIMIENTOS DE COLECTIVOS U ORGANIZACIONES SOCIALES',
    'ROBO A EMBARCACIONES': 'ROBO A EMBARCACIONES DE ESPACIOS ACUATICOS',
    'ROBO A BIENES DE INSTITUCIONES DE SALUD': 'OTROS ROBOS',
    'ROBO A BIENES DE ENTIDADES FINANCIERAS': 'ROBO A UNIDADES ECONOMICAS',
    'ROBO A VEHICULOS DE TRANSPORTE DE VALORES': 'OTROS ROBOS',
    'ROBO DE BIENES A ENTIDAD PRIVADA': 'ROBO A UNIDADES ECONOMICAS',
    'ROBO DE MOTORES EMBARCACIONES': 'ROBO A EMBARCACIONES DE ESPACIOS ACUATICOS',
    'ROBO A BANCOS - ENTIDADES  FINANCIERAS': 'ROBO A UNIDADES ECONOMICAS',
    'HURTO A PERSONAS': 'REVIEW_LABEL',
    'ROBO A  BOTES PESQUEROS – YATES- FIBRAS-VELEROS ETC.': 'ROBO A EMBARCACIONES DE ESPACIOS ACUATICOS',
    'HURTO A DOMICILIO': 'REVIEW_LABEL',
    'HURTO A MOTOS': 'REVIEW_LABEL',
    'HURTO A BIENES PUBLICOS': 'REVIEW_LABEL',
    'HURTO A CARROS': 'REVIEW_LABEL',
    'HURTO A ACCESORIOS': 'REVIEW_LABEL',
    'HURTO A BIENES DE UNIDADES ECONOMICAS': 'REVIEW_LABEL',
    'TRANSITO': 'REVIEW_LABEL',
    'ABIGEATO': 'REVIEW_LABEL',
    'HURTO A EMBARCACIONES O PARTES EN ESPACIOS ACUATICOS': 'REVIEW_LABEL',
    'HURTO A BIENES PATRIMONIALES': 'REVIEW_LABEL',
    'HURTO DE LO REQUISADO': 'REVIEW_LABEL',
    'ROBO A BIENES PATRIMONIALES': 'OTROS ROBOS',
    'HURTO DE BIENES DE USO POLICIAL O MILITAR': 'REVIEW_LABEL'}


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
    

def conectar_sql(big_data_bbdd=True, db_user='falconiel', analitica_user_password='BebuSuKO', proxy_user_password='N27a34v1', analitica_host='192.168.152.197', proxy_host='192.168.152.8'):
    
    if big_data_bbdd:
        engine_maria_db = create_engine(f"mysql+pymysql://{db_user}:{analitica_user_password}@{analitica_host}", pool_recycle=3600)
        print(f"conectando {db_user}@{analitica_host}. Espere por favor...")
    else:
        # F0s!Hu63
        engine_maria_db = create_engine(f"mysql+pymysql://{db_user}:{proxy_user_password}@{proxy_host}", pool_recycle=3600)
        print(f"conectando {db_user}@{proxy_host}. Espere por favor...")
    print(engine_maria_db.connect())
    return engine_maria_db


def format_crimestory(relato_label, dataf):
    """format_crimestory
    This function gives format to the text of Ndd by 
    chaging the letters to lower string format,  removing
    any character that is not a letter or a number, which removes 
    punctuation by the momment and removes unnecessary spaces
    at the beginning or end of the string

    Args:
        relato_label (_str_): name of the column where the text is stored
        dataf (_dataframe_): dataframe that has a column with the text of the Ndd
    """
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
            label = "OTROS ROBOS"
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
    
    
def predict_text_class_tqdm(dataf, model, label_relato= 'd_RELATO_SIAF',label_name='d_DELITOS_SEGUIMIENTO', score_label=None, words_qty_label='d_CANTIDAD_PALABRAS', threshold_words_qty=50):
    """predict_text_class_tqdm
        Classifies the given text according to model. It includes tqdm to evaluate progress
    Args:
        dataf (_type_): dataframe that contains the text to be classified
        model (_type_): model to be used. must be a hugging face pipeline pre-loaded
        label_relato (str, optional): name of the column with the text to be classified. Defaults to 'd_RELATO_SIAF'.
        label_name (str, optional): name of the new column to be returned. Defaults to 'd_DELITOS_SEGUIMIENTO'.
        words_qty_label (str, optional): name of the column with the number of words. Defaults to 'd_CANTIDAD_PALABRAS'.
        threshold_words_qty (int, optional): constant value to consider valid text to be classified. Defaults to 50.
    """
    if score_label is None:
        score_label = label_name + '_SCORE'
    tqdm.pandas()
    dataf[[label_name, score_label]] = dataf.progress_apply(lambda x: predictLabelAndScore(relato=x[label_relato], classifier=model) if x[words_qty_label] >=threshold_words_qty else ("N/A", 0), axis=1, result_type='expand')
    
    
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
        status str: name of the column that stores the state of the ndd. If 0, the ndd has not been predicted or is a new case, if 1, the ndd has been predicted 
        by modelo_seguimientos and modelo_validados previously. When 2 it was preddicted by modelo_validados. When 3 it was predicted by modelo_seguimientos. When different from 0 it can be skipped
    """
    if label_score is None:
        label_score = label_name+'_SCORE'
    tqdm.pandas()
    # HAY UN PROBLEMA EN EL APPLY, SE REQUIERE QUE ME PONGA N/A SI ERA CANTIDAD PALABRAS MENOS DE 50 Y ESTADO 0, PERO SI EL ESTADO ML ES 1, DEBE QUEDAR EL MISMO VALOR
    # dataf[[label_name, label_name+'_SCORE', status]] = dataf.progress_apply(lambda x: predictLabelAndScoreDaaS(relato=x[label_relato], classifier=model) if x[words_qty_label] >=threshold_words_qty and x[status]==0 else ("N/A", 0, 0), axis=1, result_type='expand')
    dataf[[label_name, label_score, status]] = dataf.progress_apply(lambda x: predictLabelAndScoreDaaS(relato=x[label_relato], classifier=model, actual_label=x[label_name], actual_score=x[label_score], words_qty=x[words_qty_label], threshold_words_qty=threshold_words_qty, status=x[status]), axis=1, result_type='expand')


def predict_text_class_only_new_tqdm(dataf, 
                                     model, 
                                     label_relato= 'd_RELATO_SIAF',
                                     label_name='d_DELITOS_SEGUIMIENTO', 
                                     score_label=None, 
                                     words_qty_label='d_CANTIDAD_PALABRAS', 
                                     threshold_words_qty=50, 
                                     new_ndds_list=None, 
                                     ndd_label="NDD"):
    """predict_text_class_only_new_tqdm
        Classifies the given text according to model. It includes tqdm to evaluate progress
        It predicts only new ndds after they've been compared to previous dataset from comission
    Args:
        dataf (_type_): dataframe that contains the text to be classified
        model (_type_): model to be used. must be a hugging face pipeline pre-loaded
        label_relato (str, optional): name of the column with the text to be classified. Defaults to 'd_RELATO_SIAF'.
        label_name (str, optional): name of the new column to be returned. Defaults to 'd_DELITOS_SEGUIMIENTO'.
        words_qty_label (str, optional): name of the column with the number of words. Defaults to 'd_CANTIDAD_PALABRAS'.
        threshold_words_qty (int, optional): constant value to consider valid text to be classified. Defaults to 50.
        new_ndds_list: list of new ndds to be predicted
        ndd_label: name of the column with the NDD values in the dataf 
    """
    if score_label is None:
        score_label = label_name + '_SCORE'
    tqdm.pandas()
    dataf[[label_name, score_label]] = dataf.progress_apply(
        lambda x: predictLabelAndScore(
            relato=x[label_relato], classifier=model
            ) 
            if x[words_qty_label] >=threshold_words_qty and x[ndd_label] in new_ndds_list 
            else ("N/A", 0), 
            axis=1, 
            result_type='expand')


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
    

def save_df_in_sql(name_table, dataf, index=False, where="bigData", database="reportes", chunksize=1000, db_user='falconiel', db_password='BebuSuKO', db_analitica_host='192.168.152.197', proxy_user_password='N27a34v1',  proxy_host='192.168.152.8'):
    """save_df_in_sql
    This function saves a python pandas dataframe in a SQL table
    Args:
        name_table (_str_): name of the table
        dataf (_dataframe_): dataframe with the tabular information to be saved in SQL
        index (bool, optional): If true, index is saved in sql table. Defaults to False.
        where (str, optional): Defines the database where information will be stored. Defaults to "bigData" which corresponds to the SQL server of Estadistica i.e. 192.168.152.197
        database (str, optional): Database in server where data will be stored. Defaults to "reportes".
        chunksize (int, optional): Using chunks to save data to sql. Defaults to 1000.
        db_user (str, optional): The user of the Database. Defaults to 'falconiel'.
        db_password (str, optional): Password to login in Estadistica Server. Defaults to 'BebuSuKO'.
        db_analitica_host (str, optional): Ip Address of the MySQL Estaistica server. Defaults to '192.168.152.197'.
        proxy_user_password (str, optional): Password to login in ProxyServer. Defaults to 'N27a34v1'.
        proxy_host (str, optional): Ip Address of the ProxyServer. Defaults to '192.168.152.8'.
    """        
    if where=="bigData":
        engine_maria_db = create_engine(f"mysql+pymysql://{db_user}:{db_password}@{db_analitica_host}:3306/{database}"+"?charset=utf8mb4")
    else:
        engine_maria_db = create_engine(f"mysql+pymysql://{db_user}:{proxy_user_password}@{proxy_host}:3306/{database}"+"?charset=utf8mb4")
    print("conexion con host es: {}".format(engine_maria_db.connect()))
    dataf.to_sql(name_table, engine_maria_db, if_exists='replace', index=index, chunksize=chunksize)
    with engine_maria_db.connect() as conn:
        conn.execute(f'ALTER TABLE `{database}`.`{name_table}` CONVERT TO CHARACTER SET utf8 COLLATE utf8_general_ci;')

        
def train_valid_test_sizer(dataframe_shape, proportion = (0.7,0.2,0.1)):
    """_train_valid_test_sizer_
    This function returns the sizes of the train_set, valid_set and test_set
    considering the proportions given by the tuple
    Args:
        datafram_shape (_tuple(rows, columns)_): dataset shape, it must be a tuple of the form (rows, columns)
        proportion (tuple, optional): Proportion that must sum 1. First argument is train proportion, fallowed by valid proportion and finally test proportion. Defaults to (0.7,0.2,0.1).
    """
    train_proprotion, valid_proportion, test_proportion = proportion
    if round(train_proprotion+valid_proportion+test_proportion, 1) != 1:
        print(f"proportion {proportion} does not add up to 1, try again")
        return None
    rows, cols = dataframe_shape
    TRAIN_SIZE = rows*train_proprotion
    VALID_SIZE = rows*valid_proportion
    TEST_SIZE = rows*test_proportion
    print(f"Recomended sizes are: TRAIN: {TRAIN_SIZE}, VALID: {VALID_SIZE}, TEST: {TEST_SIZE}")
    return TRAIN_SIZE, VALID_SIZE, TEST_SIZE


def seconds_to_readable_time(seconds):
    """
    Converts seconds to a human-readable time format (hours, minutes, seconds).
    Args:
        seconds (int): The input duration in seconds.
    Returns:
        str: A formatted string representing the time.
    """
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)

    # Construct the readable time format
    time_format = ""
    if hours > 0:
        time_format += f"{hours} hour{'s' if hours != 1 else ''} "
    if minutes > 0:
        time_format += f"{minutes} minute{'s' if minutes != 1 else ''} "
    if seconds > 0:
        time_format += f"{seconds} second{'s' if seconds != 1 else ''}"

    return time_format.strip()  # Remove trailing spaces


def print_robbery_kinds_qty(df, predicted_label):
    """_print_robbery_kinds_
    It allows to print summarized count of the predicted
    label and the number of empty values (i.e. NaN)
    Args:
        df (dataframe): dataframe with results of category prediction
        predicted_label (str): name of the column where categoriers were predicted 
    """
    print(f"Cantidad de categorias de {predicted_label}:\n{df[predicted_label].value_counts()}")
    print(f"Cantidad de categorias vacias {predicted_label}:{df[predicted_label].isna().sum()}")
    print(f"Total de registros: {df[predicted_label].value_counts().sum()}")


def function_unified_delitos_seguimiento(ndd, predicted_value, labeled_value, ndds_in_commision_list, estado, unified_value, origin_value):
    """create_unified_delitos_seguimiento
    This creates a unified colum delitos seguimiento named delitos_seguimiento_unified
    that keeps the value assigend by comision when ndd is in commision 
    if the ndd is not in comision or the value assigned by comision is SIN INFORMACION
    the value predicted by the model is taken
    Args:
        ndd (_str_): column label name where NDD are contained
        predicted_value (_str_): column label name where the model prediction is contained
        labeled_value (_str_): corresponds to the name of the column where we have the values assigned
                                by the comision
        ndds_in_commision_list (_list_): list that contains the NDD numbers worked by the comision
        estado (_int_): if 0 we have to look for the ndd in comision and bring its value, if 1 we skip
        unified_value (_str_): reads previous written value if exists
        origin_value (_str_): reads previous written value if exists

    Returns:
        column delitos_seguimiento_unified: column with data
    """
    # conditions:
    # if ndd in comision change if value in commision not empty or SIN INFORMACION
    # if value in comision is SIN INFORMACION change for predicted value
    if pd.isna(unified_value):
        if (ndd in ndds_in_commision_list) and (estado==0):
            if labeled_value != "SIN INFORMACION":
                return labeled_value, 'COMISION'
            else:
                return predicted_value, 'MODEL'
        else:
            return predicted_value, 'MODEL'
    else:
        return unified_value, origin_value
    

def create_delitos_seguimiento_unified(dataf, list_ndds_in_commision, ndd_col_label="NDD", predicted_delitos_col_label="delitos_seguimiento_predicted", comision_col_label="delitos_seguimiento_comision", column_label='delitos_seguimiento_unified', estado_label='ESTADO_ML_SEGUIMIENTO_UNIFIED_COMISION'):
    tqdm.pandas()
    dataf[column_label], dataf[column_label+'_origin'] = zip(*dataf.progress_apply(lambda x: function_unified_delitos_seguimiento(ndd=x[ndd_col_label],
                                                                                                                                  predicted_value=x[predicted_delitos_col_label], 
                                                                                                                                  labeled_value=x[comision_col_label],
                                                                                                                                  ndds_in_commision_list=list_ndds_in_commision,
                                                                                                                                  estado=x[estado_label],
                                                                                                                                  unified_value=x[column_label],
                                                                                                                                  origin_value=x[column_label+'_origin']), axis=1))
    
def preprocessing_delitos_seguimiento_comision(dataf, column):
    """__preprocessing_delitos_seguimiento_comision__
    To homogenize the categories found in delitos seguimiento from comision
    by removing any vowel that has accent
    Args:
        dataf (_dataframe_): dataframe with data
        column (_str_): column that contains categories with vowels accented to be changed
    """
    dataf[column] = dataf[column].str.strip()
    dataf[column] = dataf[column].str.upper()
    dataf[column] = dataf[column].str.replace('Á', 'A')
    dataf[column] = dataf[column].str.replace('É', 'E')
    dataf[column] = dataf[column].str.replace('Í', 'I')
    dataf[column] = dataf[column].str.replace('Ó', 'O')
    dataf[column] = dataf[column].str.replace('Ú', 'U')


def fix_estadoml(estadoml, estado_seguimiento, estado_validados):
    """
    Fixes ESTADO_ML output. if in a row_i, estado_seguimiento
    and estado_validados both have value of 1, then we return 1
    if only estado seguimiento or estado_validado was performed, 
    3 and 2 are returned, respectively. In any other case, the original
    value of ESTADO_ML is returned
    """
    if estado_seguimiento==1 and estado_validados==1:
        return 1
    elif estado_validados ==1:
        return 2
    elif estado_seguimiento==1:
        return 3
    else:
        return estadoml


def read_sql_comision_estadistica(database_table, 
                                  db_user,
                                  db_password,
                                  db_host):
    
    conx = conectar_sql(db_user=db_user,
                        analitica_user_password=db_password,
                        analitica_host=db_host)
    
    database, table = database_table.split('.')
    print(f"Leyendo datos desde {database}.{table}")
    query = text(f"""SELECT 
                    robos.NDD,
                    robos.Tipo_Delito_PJ as 'Tipo_Delito_PJ_comision',
                    robos.delitos_seguimiento as 'delitos_seguimiento_comision',
                    robos.delitos_validados as 'delitos_validados_comision',
                    robos.`Fecha_Incidente` as 'FechaIncidenteComision', 
                    robos.`Fecha_Registro` as 'FechaRegistroComision'
                    FROM {database}.{table} robos
                    WHERE robos.Tipo_Delito_PJ = 'ROBO';
                    """)
    return pd.read_sql(query, conx)


def read_daas_robosML(sample,
                      database_in,
                      table_in,
                      db_user,
                      db_password,
                      db_host):
    # query = "select * from `DaaS`.`robosML`"
    query = f"select * from `{database_in}`.`{table_in}`"
    if sample:
        query += " limit 1000"
    query += ";"
    query = text(query)
    daas_df = pd.read_sql(query,
                          conectar_sql(db_user=db_user,
                                       analitica_user_password=db_password,
                                       analitica_host=db_host))
    print(f"Total de registros: {daas_df.shape}")
    # dando formato al relato de la noticia del delito
    format_crimestory(relato_label='RELATO', dataf=daas_df)
    # genera la cuenta de cantidad de palabras, en columna d_CANTIDAD_PALABRAS
    words_qty(dataf=daas_df, relato_label='RELATO')
    # puede ser adecuado omitir esto luego pero podemos sobrescribir la columna \
    # cantidad de palabras original para luego hacer un drop de esa columna y no
    # afectar la estructura de la tabla original.
    daas_df['CANTIDAD_PALABRAS'] = daas_df['d_CANTIDAD_PALABRAS']
    print("Columnas del dataset {}".format(daas_df.columns))
    print(f"Características de la Cantidad de palabras\n:{daas_df.d_CANTIDAD_PALABRAS.describe()}")
    # hacer un drop de d_CANTIDAD_PALABRAS????
    return daas_df, 'RELATO'


def function_union_siaf_model(predicted_value, siaf_value, words_qty, words_qty_threshold, estado, unified_siaf_value, origin_value):
    if pd.isna(unified_siaf_value):
        if (words_qty < words_qty_threshold) and (estado==0):
            if siaf_value!= "REVIEW_LABEL":
                return siaf_value, 'SIAF'
            else:
                return predicted_value, 'MODEL'
        else:
            return predicted_value, 'MODEL'
    else:
        return unified_siaf_value, origin_value


def create_model_siaf_unified(dataf,
                              predicted_delitos_col_label="delitos_seguimiento_predicted", 
                              siaf_col_label="desagregacion_siaf", 
                              column_label='delitos_seguimiento_unified', 
                              words_qty_label='d_CANTIDAD_PALABRAS',
                              words_qty_threshold=50,
                              estado_label='ESTADO_ML_SEGUIMIENTO_UNIFIED_COMISION'):
    tqdm.pandas()
    dataf[column_label], dataf[column_label+'_origin'] = zip(*dataf.progress_apply(lambda x: function_union_siaf_model(predicted_value=x[predicted_delitos_col_label],
                                                                                                                       siaf_value=x[siaf_col_label],
                                                                                                                       words_qty_threshold=words_qty_threshold,
                                                                                                                       words_qty=x[words_qty_label],
                                                                                                                       estado=x[estado_label],
                                                                                                                       unified_siaf_value=x[column_label],
                                                                                                                       origin_value=x[column_label+'_origin']), axis=1))
    


def create_desagregacion_siaf_new_column(dataf, original_values_column,new_column_name, category_mapping):
    dataf[new_column_name] = dataf[original_values_column]
    preprocessing_delitos_seguimiento_comision(dataf=dataf,
                                               column=new_column_name)
    dataf[new_column_name] = dataf[new_column_name].replace(category_mapping)
    
