"""
https://betterprogramming.pub/deploy-interactive-real-time-data-visualizations-on-flask-with-bokeh-311239273838

https://docs.bokeh.org/en/latest/docs/user_guide/tools.html

"""

from flask import Flask, render_template
from bokeh.models import ColumnDataSource, Div, Select, Slider, TextInput, LassoSelectTool, RadioGroup, Button
from bokeh.models.callbacks import CustomJS
from bokeh.layouts import column, layout, row
from bokeh.io import curdoc
from bokeh.resources import INLINE
from bokeh.embed import components
from bokeh.plotting import figure, output_file, show
import os
import webbrowser


BUTTON_SIZE = 200



app = Flask(__name__)

@app.route('/')
def index():
    source = ColumnDataSource()
    fig = figure(
            plot_height=600,
            plot_width=720,
            tools=['lasso_select'],
    )
    fig.circle(x="x", y="y", source=source, size=8, color="color", line_color=None)
    fig.xaxis.axis_label = "X axis"
    fig.yaxis.axis_label = "Y axis"

    # select_tool = fig.select(dict(type=LassoSelectTool))[0]

    LABELS = ["Wake", "NREM", "REM"]

    wake_button = Button(label="Wake", default_size=BUTTON_SIZE)
    wake_button.js_on_click(
        CustomJS(
            args=dict(source=source),
            code="""
                var color = source.data['color']
                var idx = source.selected.indices;
                var selected_length = idx.length;
                for (var i=0; i<selected_length; i++) {
                    color[idx[i]] = '#229900';
                }
                source.selected.indices = [];
                source.change.emit();
            """,
    ))

    nrem_button = Button(label="NREM", default_size=BUTTON_SIZE)
    nrem_button.js_on_click(
        CustomJS(
            args=dict(source=source),
            code="""
                var color = source.data['color']
                var idx = source.selected.indices;
                var selected_length = idx.length;
                for (var i=0; i<selected_length; i++) {
                    color[idx[i]] = '#449944';
                }
                source.selected.indices = [];
                source.change.emit();
            """,
    ))

    rem_button = Button(label="NREM", default_size=BUTTON_SIZE)
    rem_button.js_on_click(
        CustomJS(
            args=dict(source=source),
            code="""
                var color = source.data['color']
                var idx = source.selected.indices;
                var selected_length = idx.length;
                for (var i=0; i<selected_length; i++) {
                    color[idx[i]] = '#446644';
                }
                source.selected.indices = [];
                source.change.emit();
            """,
    ))

    x = [1,2,3,4,5]
    y = [5,5,4,6,2]
    color = ["#FF9900"]*len(x)

    source.data = dict(
        x=x,
        y=y,
        color=color,
    )

    buttons = column(wake_button, nrem_button, rem_button)
    my_layout = row(buttons, fig)
    script, div = components(my_layout)
    return render_template(
            'index.html',
            plot_script=script,
            plot_div=div,
            js_resources=INLINE.render_js(),
            css_resources=INLINE.render_css(),
    ).encode(encoding='UTF-8')



if __name__ == "__main__":
    # https://stackoverflow.com/a/63216793
    # The reloader has not yet run - open the browser
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new('http://127.0.0.1:8080/')

    app.run(host="127.0.0.1", port=8080, debug=True)



###
