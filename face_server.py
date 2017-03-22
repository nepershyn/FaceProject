import sys
import flask
import numpy
import logging
import tornado.web
import tornado.ioloop
import tornado.options
import tornado.httpserver
from scipy.spatial.distance import cdist

from caffe_model import CaffeGoogleNetPlaces
from postgres_database import PostgresData
from tornado.wsgi import WSGIContainer


# Obtain the flask app object, dictionary, pickle file and logger
app = flask.Flask(__name__)
#logging.basicConfig(format=u'%(levelname)-8s [%(asctime)s] %(message)s', level=logging.DEBUG, filename='server.log')


@app.route('/get_duplicates', methods=['POST'])
def get_duplicates():
    try:
        additional_photos, cmp_list, name_list = [], [], []
        # >> Get JSON
        data = flask.request.get_json()
        photo = data['mainPhoto']
        features = google_net.get_features()


        dist = cdist(np.array(features[1]).reshape(1, 1024), features, 'braycurtis').squeeze()
        possible_duplicates = [table.data[x][:2] for x in np.where((dist>0.1)&(dist<0.175))[0]]
        duplicates = [table.data[x][:2] for x in np.where((dist<=0.1))[0]]


        table.append_row(data['REF'], data['ticket'], '0', fetures[0], features[1])

        return flask.jsonify({'Duplicates':duplicates, 'Possible_duplicates':possible_duplicates})
    except KeyError:
        logging.error('KeyError')
        return flask.jsonify({'error': 'KeyError'})
    except Exception as err:
        logging.error('Exception occurred: %s', err)
        flask.abort(404)


if __name__ == '__main__':
    google_net = CaffeGoogleNetPlaces()
    table = PostgresData()
    # >> Starting Tornado server
    tornado.options.parse_command_line()
    http_server = tornado.httpserver.HTTPServer(WSGIContainer(app))
    http_server.listen(1488)
    print("Tornado server starting successfully")
    tornado.ioloop.IOLoop.current().start()
