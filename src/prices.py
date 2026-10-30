"""确定性读取教学价格表。价格不参与文档向量检索，也不由模型编造。"""
import csv
from datetime import date as calendar_date
from decimal import Decimal, InvalidOperation
from settings import ROOT

PRICE_FILE = ROOT / 'data' / 'prices.csv'


def read_prices(path=PRICE_FILE):
    with path.open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        calendar_date.fromisoformat(row['date'])
        if row['unit'] != '元/公斤' or row['is_demo'] != 'true':
            raise ValueError('内置价格表必须明确标注教学示例，单位为元/公斤。')
        try:
            if not Decimal(row['price']).is_finite() or Decimal(row['price']) <= 0:
                raise ValueError('价格必须为正的有限数值。')
        except InvalidOperation:
            raise ValueError('价格不是有效数字。') from None
    return rows


def lookup_prices(vegetable='', date=''):
    if len(vegetable) > 30:
        raise ValueError('蔬菜名称最多 30 字。')
    rows = [r for r in read_prices() if not vegetable or r['vegetable'] == vegetable.strip()]
    if date:
        calendar_date.fromisoformat(date)
    selected_date = date or max((r['date'] for r in rows), default='')
    selected = [r for r in rows if r['date'] == selected_date]
    return {'is_demo': True, 'source': '项目自编教学价格表 data/prices.csv',
            'as_of': selected_date, 'date_policy': '指定日期' if date else '该品种最新有记录日期',
            'unit': '元/公斤', 'records': selected,
            'notice': '虚构教学数据，不是实时行情；缺少指定日期时不自动换日。' +
                      ('' if selected else ' 没有匹配记录。')}
