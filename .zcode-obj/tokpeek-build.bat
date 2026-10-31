@echo off
call "D:\VisualStudio\VC\Auxiliary\Build\vcvars64.bat" >nul
cl /nologo /EHsc /std:c++20 /utf-8 /MD /I F:\Strata-dev\include F:\Strata-dev\tests\program\tokpeek.cpp /link F:\Strata-dev\build\strata_grammar.lib /OUT:F:\Strata-dev\.zcode-obj\tokpeek.exe
