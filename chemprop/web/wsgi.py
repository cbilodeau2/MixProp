"""
Runs the web interface version of Chemprop.
Designed to be used for production only, along with Gunicorn.
"""
from chemprop.web.app import app, db


def build_app(*args, **kwargs):
    db.init_app(app)
    if True:
        with app.app_context():
            db.init_db()
            print("-- INITIALIZED DATABASE --")
        
    app.config['DEMO'] = kwargs.get('demo', False)

    return app
<<<<<<< HEAD:web/wsgi.py
    
=======
>>>>>>> 81d21506373485c15c95301a2b3767340d6fcf24:chemprop/web/wsgi.py
