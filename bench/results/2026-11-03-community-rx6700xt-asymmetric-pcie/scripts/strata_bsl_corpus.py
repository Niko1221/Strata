"""Synthetic BSL reading benchmark; not deployed or executed on a 1C platform."""
import json
def build_bsl_case(helper_count=500):
 first='''Функция ИтогЗаказа(Количество, Цена, ПроцентСкидки) Экспорт
    СуммаДоСкидки = Количество * Цена;
    Возврат НачислитьНалог(ПрименитьСкидку(СуммаДоСкидки, ПроцентСкидки));
КонецФункции
'''
 middle='''Функция ПрименитьСкидку(СуммаДоСкидки, ПроцентСкидки)
    Возврат СуммаДоСкидки - ПроцентСкидки;
КонецФункции
'''
 last='''Функция НачислитьНалог(СуммаПослеСкидки)
    Возврат СуммаПослеСкидки * 1.2;
КонецФункции
'''
 blocks=[first]
 for i in range(helper_count):
  if i==helper_count//2:blocks.append(middle)
  blocks.append(f'Функция АрхивныйРасчет{i}(АргументРасчета)\n    Возврат АргументРасчета + {i%97};\nКонецФункции\n')
 blocks.append(last)
 code='\n'.join(blocks)
 return {'id':f'bsl-discount-{helper_count}','system':'Ты анализируешь синтетический модуль BSL. Верни только JSON, без Markdown. Не выполняй код и не обращайся к базе.',
 'messages':[{'role':'user','content':'Бизнес-правило: скидка задаётся процентом, затем начисляется налог 20%. Для вызова ИтогЗаказа(10, 100, 15) найди ошибку и верни поля actual, expected, faulty_function, corrected_return, call_chain (массив функций в порядке вычисления).\n\n'+code}],
 'expected':{'actual':1182,'expected':1020,'faulty_function':'ПрименитьСкидку','corrected_return':'Возврат СуммаДоСкидки * (1 - ПроцентСкидки / 100);','call_chain':['ИтогЗаказа','ПрименитьСкидку','НачислитьНалог']},
 'platform_executed':False,'token_count':None,'helper_count':helper_count}
def score(case,text):
 try:
  answer=json.loads(text)
  keys=['actual','expected','faulty_function','call_chain']
  fields={k:answer.get(k)==case['expected'][k] for k in keys}
  return {'fields':fields,'core_pass':all(fields.values()),'fix_requires_review':True,'proposed_fix':answer.get('corrected_return')}
 except (ValueError,AttributeError):return {'core_pass':False,'json_valid':False}
if __name__=='__main__':
 c=build_bsl_case(10);assert score(c,json.dumps(c['expected']))['core_pass'];assert not score(c,'{}')['core_pass']
 assert (10*100-15)*1.2==1182 and 10*100*(1-15/100)*1.2==1020
 print('BSL_CASE_ORACLE_CHECK_PASS; 1C runtime not executed')
