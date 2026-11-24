/* All the interface strings of one language (ar): the base file and the 18 add-on files, merged in the order the app has always used (a later file wins).
   Loaded the first time the language is used (see loadCatalog in i18n.ts); English stays in the main file because the English text is the key. */
import base from '../locales/ar.json';
import tune from '../locales/tune-ar.json';
import tools from '../locales/tools-ar.json';
import fix from '../locales/fix-ar.json';
import gh from '../locales/gh-ar.json';
import ui from '../locales/ui-ar.json';
import engine from '../locales/engine-ar.json';
import pick from '../locales/pick-ar.json';
import df2 from '../locales/df2-ar.json';
import upd from '../locales/upd-ar.json';
import r3 from '../locales/r3-ar.json';
import pic from '../locales/pic-ar.json';
import gui from '../locales/gui-ar.json';
import sav from '../locales/sav-ar.json';
import net from '../locales/net-ar.json';
import dmg from '../locales/dmg-ar.json';
import guic from '../locales/guic-ar.json';
import fixa from '../locales/fixa-ar.json';
import gtui from '../locales/gtui-ar.json';
export default {...base,...tune,...tools,...fix,...gh,...ui,...engine,...pick,...df2,...upd,...r3,...pic,...gui,...sav,...net,...dmg,...guic,...fixa,...gtui} as Record<string,string>;
