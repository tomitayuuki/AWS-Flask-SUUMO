from apps import db


# class Shohin(db.Model):
#     __tablename__ = 'Shohin'
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.Text)
#     price = db.Column(db.Integer)

class SuumoRecomm(db.Model):
    __tablename__ = 'suumo_recomm'
    id = db.Column(db.Integer, primary_key=True)
    nearest_station = db.Column(db.Float)
    area = db.Column(db.Float)
    age = db.Column(db.Integer)
    top_floor = db.Column(db.Integer)
    rent_floor = db.Column(db.Integer)
    land_price = db.Column(db.Float)
    rent = db.Column(db.Float)
    url = db.Column(db.String)

class LandPrice(db.Model):
    __tablename__ = 'land_price'
    id = db.Column(db.Integer, primary_key = True)
    ward = db.Column(db.String)
    land_price = db.Column(db.Integer)

class TransformFloorPlanArea(db.Model):
    __tablename__ = 'transform_floor_plan_area'
    id = db.Column(db.Integer, primary_key = True)
    floor_plan = db.Column(db.String)
    area = db.Column(db.Float)

class TableCategory(db.Model):
    __tablename__ = 'table_cstegory'
    id = db.Column(db.Integer, primary_key=True)
    category = db.Column(db.String)

class TableFloorPlan(db.Model):
    __tablename__ = 'table_floor_plan'
    id = db.Column(db.Integer, primary_key=True)
    floor_plan = db.Column(db.String)

class TableLine(db.Model):
    __tablename__ = 'table_line'
    id = db.Column(db.Integer, primary_key=True)
    line = db.Column(db.String)

class TableStation(db.Model):
    __tablename__ = 'table_station'
    id = db.Column(db.Integer, primary_key=True)
    station = db.Column(db.String)

class TableWard(db.Model):
    __tablename__ = 'table_ward'
    id = db.Column(db.Integer, primary_key=True)
    ward = db.Column(db.String)
