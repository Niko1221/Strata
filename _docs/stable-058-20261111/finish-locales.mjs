import { readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { resolve, basename } from 'node:path';
import { execFileSync } from 'node:child_process';
const root = resolve(import.meta.dirname, '../../app/host/src/locales');
function save(file, data) {
 const original = execFileSync('git', ['show', 'HEAD:apps/host/src/locales/' + basename(file)], {cwd: resolve(import.meta.dirname, '../..'), encoding:'utf8'});
 const indent = /\n([ \t]+)"/.exec(original)?.[1] ?? 2;
 writeFileSync(file, JSON.stringify(data, null, indent) + '\n');
}
const copy = {
 en: ['Workspace', 'Move to trash'], zh: ['工作区', '移到废纸篓'], ja: ['ワークスペース', 'ごみ箱に移動'],
 de: ['Arbeitsbereich', 'In den Papierkorb verschieben'], es: ['Espacio de trabajo', 'Mover a la papelera'],
 fr: ['Espace de travail', 'Déplacer vers la corbeille'], ko: ['작업 공간', '휴지통으로 이동'],
 pt: ['Espaço de trabalho', 'Mover para a lixeira'], ru: ['Рабочая область', 'Переместить в корзину'],
};
for (const [language, values] of Object.entries(copy)) {
 const file = resolve(root, language + '.json'), data = JSON.parse(readFileSync(file, 'utf8'));
 data.wc05801 = values[0]; data.wc05802 = values[1];
 if (language === 'fr') for (const key of Object.keys(data)) data[key] = data[key].replace(/ (?=[?!:;])/g, '\u00a0');
 save(file, data);
}
for (const name of readdirSync(root).filter(name => name === 'fr.json' || name.endsWith('-fr.json'))) {
 const file = resolve(root, name), data = JSON.parse(readFileSync(file, 'utf8'));
 const english = JSON.parse(readFileSync(resolve(root, name.replace('fr.json', 'en.json')), 'utf8'));
 for (const key of Object.keys(data)) if (data[key] !== english[key]) data[key] = data[key].replace(/ (?=[?!:;])/g, '\u00a0');
 save(file, data);
}
