# for converting data into numbers
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
import pandas as pd
df = pd.read_csv('decision_data.csv')
# print(df)


inputs = df.drop('Label', axis='columns')
target = df.drop(['Color', 'Diameter'], axis='columns')
# print(inputs)
# print(target)


# to convert data into numbers
le_color = LabelEncoder()
le_dia = LabelEncoder()
le_label = LabelEncoder()

# adding converted columns
inputs['color_n'] = le_color.fit_transform(inputs['Color'])
inputs['diameter_n'] = le_dia.fit_transform(inputs['Diameter'])
target['label_n'] = le_label.fit_transform(target['Label'])
# print(inputs)
# print(target)


# inputs with converted columns only
inputs_n = inputs.drop(['Color', 'Diameter'], axis='columns')
target_n = target.drop(['Label'], axis='columns')
# print(inputs_n)
# print(target_n)

model = tree.DecisionTreeClassifier()
model.fit(inputs_n, target)
# print(model)

print(model.score(inputs_n, target))

print(model.predict([2, 1]))
