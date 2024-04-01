# TODO 1 

It is possible that we may need to make a new engineering of all of this due to the increasing size of robos

## SQL-CHUNKS


    import pandas as pd
    from sqlalchemy import create_engine

    # Create a connection to the database
    engine = create_engine('mysql://username:password@localhost/dbname')

    chunksize = 50000  # depends on your available memory
    offset = 0
    while True:
        sql = "SELECT * FROM tablename LIMIT %d OFFSET %d" % (chunksize, offset)
        df = pd.read_sql(sql, engine)
        if len(df) == 0:
            break  # All rows have been read
        else:
            offset += chunksize
        # Apply your model to the data chunk
        predictions = model.predict(df)
        # TODO: Save/store the predictions

## Dask

I have tried some options with dask but it is required that the table 
has a numeric primary key. You cannot use NDD as a index col


    import dask.dataframe as dd

    # This loads the data into a Dask DataFrame
    ddf = dd.read_sql_table('tablename', 'mysql://username:password@localhost/dbname', index_col='id')

    # This doesn't load the data into memory
    predictions = model.predict(ddf)

    # This will compute the result and load it into memory
    predictions = predictions.compute()

# TODO -2 

Cuando se ha realizado la prediccion del modelo y se ejecuta el programa nuevamente, los valores con estado distinto de 0 se saltan y se quedan llenos con nulos. Esto afecta la consolidacion de la tabla ya que borra la informacion. Una mejora seria que si el estado es distinto de 0 y concretamente 1,2, o 3, se vuelva a colocar
el valor que ya esta en la tabla de robos para no perder el dato. Esta es una reparacion urgente


