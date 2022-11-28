from flask import render_template, request, redirect, url_for
from apps import app, db
import lightgbm as lgb
import pandas as pd
import pickle
import numpy as np
from copy import copy
from apps.models import *

@app.before_first_request
def init():
    db.create_all()

    if db.session.query(SuumoRecomm).count() == 0:

        filepath = "./apps/data/suumo_recomm.csv"
        table = pd.read_csv(filepath)

        for index, rows in table.iterrows():
            suumo_recomm = SuumoRecomm(
                nearest_station = rows['最寄駅からの距離'],
                area = rows['面積'],
                age = rows['築年数'],
                top_floor = rows['地上'],
                rent_floor = rows['階'],
                land_price = rows['地価'],
                rent = rows['家賃'],
                url = rows['url'],
            )
            db.session.add(suumo_recomm)
            db.session.commit()

        filepath = "./apps/data/land_price.csv"
        table = pd.read_csv(filepath)

        for index, rows in table.iterrows():
            land_price = LandPrice(
                ward=rows['ward'],
                land_price=rows['地価'],
                )
            db.session.add(land_price)
            db.session.commit()

        filepath = "./apps/data/transform_floor_plan_area.csv"
        table = pd.read_csv(filepath)

        for index, rows in table.iterrows():
            transform_floor_plan_area = TransformFloorPlanArea(
                floor_plan=rows['floor_plan'],
                area=rows['面積'],
                )
            db.session.add(transform_floor_plan_area)
            db.session.commit()

        filepath = "./apps/data/table_category.csv"
        table = pd.read_csv(filepath)

        for index, rows in table.iterrows():
            table_category = TableCategory(
                category=rows['name']
                )
            db.session.add(table_category)
            db.session.commit()

        filepath = "./apps/data/table_floor_plan.csv"
        table = pd.read_csv(filepath)

        for index, rows in table.iterrows():
            table_floor_plan = TableFloorPlan(
                floor_plan=rows['name'],
                )
            db.session.add(table_floor_plan)
            db.session.commit()

        filepath = "./apps/data/table_line.csv"
        table = pd.read_csv(filepath)

        for index, rows in table.iterrows():
            table_line = TableLine(
                line=rows['name'],
                )
            db.session.add(table_line)
            db.session.commit()

        filepath = "./apps/data/table_station.csv"
        table = pd.read_csv(filepath)

        for index, rows in table.iterrows():
            table_station = TableStation(
                station=rows['name'],
                )
            db.session.add(table_station)
            db.session.commit()

        filepath = "./apps/data/table_ward.csv"
        table = pd.read_csv(filepath)

        for index, rows in table.iterrows():
            table_ward = TableWard(
                ward=rows['name'],
                )
            db.session.add(table_ward)
            db.session.commit()


# @app.route('/', methods=['GET'])
# def index():
#     datas = Shohin.query.all()
#     return render_template('index.html', lists = datas)

# @app.route('/result', methods=['POST'])
# def insert():
#     name_txt = request.form['name']
#     price_txt = request.form['price']
#     shohin = Shohin(name = name_txt, price = price_txt)

#     db.session.add(shohin)
#     db.session.commit()

#     return redirect('/')




@app.route("/")
def index():
    return redirect(url_for("predict"))

@app.route("/predict")
def predict():
    table_category = get_table_to_df(TableCategory)
    table_ward = get_table_to_df(TableWard)
    table_line = get_table_to_df(TableLine)
    table_station = get_table_to_df(TableStation)
    table_floor_plan = get_table_to_df(TableFloorPlan)
    return render_template("predict.html",
                            table_category=table_category,
                            table_ward=table_ward,
                            table_line=table_line,
                            table_station=table_station,
                            table_floor_plan=table_floor_plan
                            )

@app.route("/predict_result", methods=["post"])
def predict_result():

    # formの入力をdictで管理
    predict_data = {
        "nearest_station": float(request.form["nearest_station"]),
        "age": int(request.form["age"]),
        "top_floor": int(request.form["top_floor"]),
        "area": float(request.form["area"]),
        "category": int(request.form["category"]),
        "ward": int(request.form["ward"]),
        "line": int(request.form["line"]),
        "station": int(request.form["station"]),
        "floor_plan": int(request.form["floor_plan"]),
    }

    # LightGBM
    clf = pickle.load(open('apps/data/trained_LGBM.pkl', 'rb'))
    before_predict=np.array(list(predict_data.values())).reshape(1,-1)
    pred = clf.predict(before_predict)[0]
    pred_rent = round(10**pred, 1)

    # predict_dataの値を書き直す
    table_category = get_table_to_df(TableCategory)
    table_ward = get_table_to_df(TableWard)
    table_line = get_table_to_df(TableLine)
    table_station = get_table_to_df(TableStation)
    table_floor_plan = get_table_to_df(TableFloorPlan)
    for num, (key, item) in enumerate(predict_data.items()):
        print(key,item)
        if num < 4:
            continue
        exec(f"row_num = list(zip(*np.where(table_{key}['id'] == {item})))[0][0]")
        exec(f"predict_data['{key}'] = table_{key}.loc[row_num,'{key}']")

    # 日本語に直すための対応表
    eng_jap = {
        "nearest_station": "最寄駅からの距離",
        "age": "築年数",
        "top_floor": "総階数",
        "area": "専有面積",
        "category": "カテゴリ",
        "ward": "市区町村",
        "line": "路線",
        "station": "駅",
        "floor_plan": "間取り",
    }

    return render_template("predict_result.html", predict_data=predict_data, pred_rent=pred_rent, eng_jap=eng_jap)

@app.route("/recommend")
def recommend():
        transform_floor_plan_area = get_table_to_df(TransformFloorPlanArea)
        land_price = get_table_to_df(LandPrice)
        return render_template("recommend.html",
            transform_floor_plan_area=transform_floor_plan_area,
            land_price = land_price)

@app.route("/recommend_result", methods=["post"])
def recommend_result():
    # フォームデータを格納
    ideal_rent = {
        "rent": float(request.form["rent"]),
        "ward": request.form["ward"],
        "time_to_station": int(request.form["time_to_station"]),
        "floor_plan": request.form["floor_plan"],
        "top_floor": int(request.form["top_floor"]),
        "rent_floor": int(request.form["rent_floor"]),
        "age": int(request.form["age"]),
    }
    print(ideal_rent)

    # 駅徒歩を変換
    ideal_rent['nearest_station'] = ideal_rent['time_to_station']*100

    # 間取りを変換
    # その場で計算してもいいけどテーブルを用意してもいいかもね
    transform_floor_plan_area = get_table_to_df(TransformFloorPlanArea)
    condition = transform_floor_plan_area['floor_plan'] == ideal_rent['floor_plan']
    ideal_rent['area'] = transform_floor_plan_area.loc[condition,'area'].values[0]

    # 地価を変換
    # その場で計算してもいいけどテーブルを用意してもいいかもね
    land_price = get_table_to_df(LandPrice)
    condition = land_price['ward'] == ideal_rent['ward']
    ideal_rent['land_price'] = land_price.loc[condition,'land_price'].values[0]

    # 家賃を変換
    ideal_rent['rent'] = np.log10(ideal_rent['rent'])

    # 整形
    df_ideal = pd.DataFrame(ideal_rent.values()).T
    df_ideal.columns = ideal_rent.keys()

    # 物件データ
    suumo_recomm = get_table_to_df(SuumoRecomm)
    # 変数
    features = [
        'nearest_station',
        'area',
        'age',
        'top_floor',
        'rent_floor',
        'land_price',
        'rent',
    ]

    # 標準化
    from sklearn.preprocessing import StandardScaler
    ss = StandardScaler()
    ss.fit(suumo_recomm[features])
    suumo_recomm_ss = copy(suumo_recomm)
    suumo_recomm_ss[features] = ss.transform(suumo_recomm[features])

    # 標準化を適用
    df_ideal_ss = copy(df_ideal)
    df_ideal_ss[features] = ss.transform(df_ideal[features])

    # コサイン類似度
    def cos_sim(v1,v2):
        dot = np.dot(v1,v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        sim = dot/(norm1*norm2)
        return sim

    # 物件の類似度を計算
    vec_sim = suumo_recomm_ss.apply(lambda x: cos_sim(x[features],df_ideal_ss.loc[0,features]),axis=1)

    # 一番類似している物件を3件
    num = vec_sim.nlargest(3).index.tolist()
    max_ = vec_sim.nlargest(3).values.tolist()

    # 変形したデータをもとに戻す
    # フォームデータ
    ideal_rent['rent'] = (10**ideal_rent['rent']).round(1)

    # 物件データ
    recomm_rent = ss.inverse_transform(suumo_recomm_ss.loc[num,features])
    recomm_rent = pd.DataFrame(recomm_rent,index=num, columns=features)

    # 面積→間取り
    recomm_rent['floor_plan'] = recomm_rent['area'].map(lambda x: transform_floor_plan_area.loc[transform_floor_plan_area['area'] == x,'floor_plan'].values[0])

    # 地価→区
    recomm_rent['ward'] = recomm_rent['land_price'].map(lambda x: land_price.loc[land_price['land_price'] == round(x),'ward'].values[0])

    # 最寄駅からの距離→駅徒歩
    recomm_rent['time_to_station'] = (recomm_rent['nearest_station']/100).astype(int)

    # 家賃をもとに戻す
    recomm_rent['rent'] = (10**(recomm_rent['rent'])).round(1)

    # 不要な要素を消す
    del ideal_rent["nearest_station"],ideal_rent["land_price"],ideal_rent["area"]
    recomm_rent.drop(["nearest_station","land_price","area"], axis=1, inplace=True)

    # 順番調整
    recomm_rent = recomm_rent[ideal_rent.keys()]

    # 型変換
    recomm_rent["age"] = recomm_rent["age"].astype(int)
    recomm_rent["top_floor"] = recomm_rent["top_floor"].astype(int)
    recomm_rent["rent_floor"] = recomm_rent["rent_floor"].astype(int)

    # 類似度を与える
    recomm_rent['similarity'] = np.array(max_).reshape(-1,1)
    recomm_rent["similarity"] = recomm_rent["similarity"].round(3)

    # urlを与える
    recomm_rent['url'] = suumo_recomm_ss.loc[num,'url']

    return render_template("recommend_result.html",
        ideal_rent=ideal_rent,recomm_rent=recomm_rent)

@app.route("/sql")
def sql():

    table_ward = get_table_to_df(TableCategory)
    print(table_ward)

    return "コンソール"

def get_table_to_df(tablename):
    data = db.session.query(tablename).all()
    table = []
    for ward in data:
        columns = []
        for column in tablename.__table__.c.keys():
            exec(f"columns.append(ward.{column})")
        table.append(columns)
    df = pd.DataFrame(table,columns=tablename.__table__.c.keys())
    return df
