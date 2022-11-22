import base64
import gc
import json
import uuid
from io import BytesIO

import numpy as np
import torch
from PIL import Image
from flask import Flask, jsonify, abort
from flask import request
from flask_sqlalchemy import SQLAlchemy

import detector
from utils import normalize_image


# Initialize SQLAlchemy
db = SQLAlchemy()
app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///project.db"
db.init_app(app)

import datetime
from sqlalchemy import Column, Integer, Text, DateTime


class Entry(db.Model):
    id = Column(Integer, primary_key=True)
    uuid = Column(Text)  # UUID, Used to categorize entries
    image = Column(Text)
    segments = Column(Text)
    percentage = Column(Integer, nullable=True)
    most_significant_detection = Column(Text)
    most_significant_area = Column(Integer)
    date_created = Column(DateTime, default=datetime.datetime.utcnow)


with app.app_context():
    db.create_all()

d = detector.Detector()


@app.route('/uuid', methods=['GET'])
def generate_uuid():
    return str(uuid.uuid4())


@app.route("/", methods=["POST"])
def predict():
    if request.files.get('image') is not None:
        file = request.files['image'].read()
        input_image = normalize_image(file)
    else:
        file = request.json.get('image')
        file = file.replace('data:image/jpeg;base64,', '')
        file = base64.b64decode(file)
        input_image = Image.open(BytesIO(file))
        input_image = np.array(input_image)

    panoptic_seg, segments_info, out, metadata = d.predict(input_image)

    # Convert to base64
    output_image = Image.fromarray(out, 'RGB')
    buffered = BytesIO()
    output_image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue())

    most_significant_detection = (None, 0)
    for i in range(len(segments_info)):
        # Add category_title to segments_info
        if segments_info[i]['isthing']:
            segments_info[i]['category_title'] = metadata.thing_classes[segments_info[i]['category_id']]
        else:
            segments_info[i]['category_title'] = metadata.stuff_classes[segments_info[i]['category_id']]

        if segments_info[i]['category_title'] not in ['wall', 'wall-brick', 'wall-concrete', 'wall-stone', 'wall-wood']:
            continue
        # Set most significant detection
        if segments_info[i]['area'] > most_significant_detection[1]:
            most_significant_detection = (segments_info[i]['category_title'], segments_info[i]['area'])

    # Calculate Percentage
    if request.form.get('percentage', None):
        percentage = request.form.get('percentage')
    elif request.json.get('percentage', None):
        percentage = request.json.get('percentage')
    else:
        # previous = get_first_entry_for(request.form.get('uuid', request.json.get('uuid')))
        # print(previous)
        # if not previous or previous.percentage is None:
        #     abort(400)
        # percentage = int(most_significant_detection[1] / (previous.most_significant_area / previous.percentage))
        percentage = int(most_significant_detection[1] / ((output_image.size[0] * output_image.size[1]) * 0.8) * 100)
    # Save to Database
    
    entry = Entry(uuid=request.form.get('uuid', request.json.get('uuid')), image=encoded_image.decode('utf-8'),
                  segments=json.dumps(segments_info),
                  percentage=percentage,
                  most_significant_detection=most_significant_detection[0],
                  most_significant_area=most_significant_detection[1])
    db.session.add(entry)
    db.session.commit()

    # Garbage collection
    gc.collect()
    torch.cuda.empty_cache()

    entities = Entry.query.filter_by(uuid=request.form.get('uuid')).all()

    return {
        'image': encoded_image.decode("utf-8"),
        'segments': segments_info,
        'entries': [{
            'id': entity.id,
            'uuid': entity.uuid,
            'percentage': entity.percentage,
            'most_significant_detection': entity.most_significant_detection,
            'most_significant_area': entity.most_significant_area,
            'date_created': int(entity.date_created.timestamp())
        } for entity in entities]
    }


# Get all entities for a specific uuid through /entities/{uuid}
@app.route("/entities/<uuid>", methods=["GET"])
def get_all_entries_for(uuid):
    
    return jsonify([{
        'id': entity.id,
        'uuid': entity.uuid,
        'percentage': entity.percentage,
        'most_significant_detection': entity.most_significant_detection,
        'most_significant_area': entity.most_significant_area,
        "date_created": entity.date_created.strftime("%Y-%m-%d %H:%M:%S"),
    } for entity in Entry.query.filter_by(uuid=uuid).all()])


@app.route("/entities/<uuid>/<id>", methods=["GET"])
def get_entity(uuid, id):
    
    entity = Entry.query.filter_by(uuid=uuid, id=id).first()
    if entity:
        return jsonify({
            'id': entity.id,
            'uuid': entity.uuid,
            'image': 'data:image/jpeg;base64,' + entity.image,
            'percentage': entity.percentage,
            'most_significant_detection': entity.most_significant_detection,
            'most_significant_area': entity.most_significant_area,
            'date_created': int(entity.date_created.timestamp()),
            'segments': json.loads(entity.segments),
        })
    else:
        abort(404)

@app.route("/entities/<uuid>/check", methods=['GET'])
def check_entry_exists(uuid):
    
    if not Entry.query.filter_by(uuid=uuid).first():
        return '', 404
    return '', 204

# Get Latest entry for a given uuid
def get_first_entry_for(uuid):
    
    return Entry.query.filter_by(uuid=uuid).first()


if __name__ == '__main__':
    app.run()
