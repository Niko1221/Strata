/* All the interface strings of one language (it): the base file and the 16 add-on files, merged in the order the app has always used (a later file wins).
 Loaded the first time the language is used (see loadCatalog in i18n.ts); English stays in the main file because the English text is the key. */
import base from '../locales/it.json';
import tune from '../locales/tune-it.json';
import tools from '../locales/tools-it.json';
import fix from '../locales/fix-it.json';
import gh from '../locales/gh-it.json';
import ui from '../locales/ui-it.json';
import engine from '../locales/engine-it.json';
import pick from '../locales/pick-it.json';
import df2 from '../locales/df2-it.json';
import upd from '../locales/upd-it.json';
import r3 from '../locales/r3-it.json';
import pic from '../locales/pic-it.json';
import gui from '../locales/gui-it.json';
import sav from '../locales/sav-it.json';
import net from '../locales/net-it.json';
export default {...base,...tune,...tools,...fix,...gh,...ui,...engine,...pick,...df2,...upd,...r3,...pic,...gui,...sav,...net,...dmg,...guic,...fixa,...gtui} as Record<string,string>;
import dmg from '../locales/dmg-it.json';
import guic from '../locales/guic-it.json';
import fixa from '../locales/fixa-it.json';
import gtui from '../locales/gtui-it.json';
