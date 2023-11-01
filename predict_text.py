from transformers import AutoTokenizer, pipeline
PATH_MODEL = '/home/falconiel/ML_Models/robbery_tf20221113'
model_ckpt = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
classifier = pipeline("text-classification", model=PATH_MODEL, tokenizer=tokenizer)
texto = "el día de ayer 18 de mayo del 2022 a eso de las 21h35 ha llegado a mi casa maribel alexandra chugchilan vaca con quien tengo un hijo de 1 años 8 meses y vive en la parroquia once de noviembre quien al momento de verificar que están solo mis dos hijos de 12 y 10 de edad ella ingresa a la casa ha estado conversando con mis hijos hasta las 10 de la noche en que ha llegado un camión a la casa y para que mi hija no avise maribel alexandra inmediatamente le ha quitado a la fuerza el teléfono a mi hija de 12 años y como han estado en un cuarto mi hija se ha asustado porque le ha dicho en forma amenazante ahí te quedas les ha cerrado la puerta y no le ha dejado salir después de unos 15 minutos le deja el teléfono en un mueble y maribel se ha llevado en el camión que ha llegado presuntamente con los hermanos las cosas de la casa como son la cocina y refrigeradora marca indurama lavadora marca whirlpool un licuadora marca oster una máquina de coser marca singer además la mitad de los muebles de  sala enceres de cocina como ollas platos cucharas además una potencia y  bajos que tenía para el carro marca kenwood parlantes marca pionner una cama de 2 plazas con su colchón un armario grade con todo lo que tenía en el interior tdos cilindros de gas 3 botellas de wiski objetos de limpieza y aseo personal de míos y de mis hijos"

y_hat = classifier(texto, truncation=True, return_all_scores=True)

   

