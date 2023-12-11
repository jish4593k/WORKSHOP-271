import tornado.ioloop
import tornado.web
from tornado.options import define, options

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import datasets
from sklearn.model_selection import train_test_split
import seaborn as sns
import scipy.stats as stats

# Define command-line options
define("port", default=8888, help="run on the given port", type=int)

# Load a sample dataset for demonstration
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Build a simple neural network using Keras
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Some data for seaborn plotting
data = np.random.randn(1000)

# Define Tornado RequestHandlers
class HelloHandler(tornado.web.RequestHandler):
    def get(self):
        self.write('Hello, Tornado 🌪️!')

class PostHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("<h1>This is Post 1 ✍️</h1>")

class HomeHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("home.html")

class WeatherHandler(tornado.web.RequestHandler):
    def get(self):
        degree = int(self.get_argument("degree"))
        output = "Hot ☀️!" if degree > 20 else "Cold 🌦️"
        drink = "Have some Beer 🍺!" if degree > 20 else "You need a hot beverage ☕"
        self.render("weather.html", output=output, drink=drink)

class MLHandler(tornado.web.RequestHandler):
    def get(self):
       n
        sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])
        prediction = model.predict(sample_input)
        self.write(f"Model Prediction: {prediction}")

class PlotHandler(tornado.web.RequestHandler):
    def get(self):
     
        sns.histplot(data, kde=True)
        self.write("Histogram plotted using seaborn.")
        self.finish()

class StatsHandler(tornado.web.RequestHandler):
    def get(self):
     
        mean, p_value = stats.ttest_1samp(data, 0)
        self.write(f"Mean: {mean}, p-value: {p_value}")



if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = make_app()
    app.listen(options.port)
    print(f'🌐 Server is listening on localhost on port {options.port}')
    tornado.ioloop.IOLoop.current().start()
