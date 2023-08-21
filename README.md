<h2>Нейросеть, распознающая рукописные цифры</h2>
  <p>Автор: Андрей Бочков. Это модифицированная версия программ от автора "Хауди Хо™ - Просто о мире IT!"</p>
  <h3>Как открыть:</h3>
  <ol>
    <li>Скачать и установить Python IDLE 3.10 или выше. Для работы программ также понадобятся библиотеки numpy и matplotlib (консольная команда "pip install имя_библиотеки" поможет установить их).</li>
    <li>Двойным щелчком в файловом менеджере открыть файл save.py, чтобы создать новую нейросеть, обучить её и сохранить в файл "neural.network". load.py нужен для работы с уже сохранённой сеткой.</li>
    <li>При запуске программ читать инструкцию, выводящуюся в открывающемся окне</li>
  </ol>
<h2>Советы к эксплуатации</h2>
  <ul>
    <li>Нейросеть при низких настройках бывает <b>очень</b> неточна. Да, это очевидно, но, видимо, не для всех. Ставьте значение запоминания не меньше среднего во избежание "шестёрок, которые восемь" и "единиц, которые семь".</li>
    <li>Слишком большие значения способности запоминания нейросети также стоит избегать, если в вашем компьютере процессор не мощности "супер"</li>
    <li>Если вы собираетесь рисовать свои картинки, то размещайте цифру в центре. В том пакете данных, на котором обучается нейросеть, все цифры стоят по центру, поэтому столкнувшись с нетрадиционным заданием, нейросеть ответит скорее случайно, чем правильно.</li>
  </ul>