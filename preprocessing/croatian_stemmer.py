#!/usr/bin/python2
#-*-coding:utf-8-*-
#
#    Simple stemmer for Croatian v0.1
#    Copyright 2012 Nikola Ljubešić and Ivan Pandžić
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published
#    by the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import re
import sys
import pkg_resources as pkg


def istakniSlogotvornoR(niz):
    return re.sub(r'(^|[^aeiou])r($|[^aeiou])', r'\1R\2', niz)


def imaSamoglasnik(niz):
    if re.search(r'[aeiouR]', istakniSlogotvornoR(niz)) is None:
        return False
    else:
        return True


def transform(token):
    for source_word, transformed_word in transformations:
        if token.endswith(source_word):
            return token[:-len(source_word)]+transformed_word
    return token


def stem_token(token):
    for rule in rules:
        rule_match = rule.match(token)
        if rule_match is not None:
            if imaSamoglasnik(rule_match.group(1)) and len(rule_match.group(1)) > 1:
                return rule_match.group(1)
    return token


def get_stop_words(stop_stream):
    stop_words = []
    content = stop_stream.read().decode('iso8859_2').encode('utf-8').split('\r\n')

    for line in content:
        if len(line) == 0:
            continue
        if line[0] != ';':
            stop_words.append(line.split(';')[0].strip())
    return [stop.decode('utf-8') for stop in stop_words]


def get_rule(rule_stream):
    return [re.compile(r'^('+osnova+')('+nastavak+r')$') for osnova, nastavak in
            [e.decode('utf8').strip().split(' ') for e in rule_stream]]


def get_transformations(transformation_stream):
    return [e.decode('utf8').strip().split('\t') for e in transformation_stream]


def stem_document(document, keep_stop_words=False):
    stemmed = []
    for token in re.findall(r'\w+', document.decode('utf8'), re.UNICODE):
        if token.lower() in stop:
            if keep_stop_words:
                stemmed.append(token.lower().encode('utf8'))
            continue
        stemmed.append(stem_token(transform(token.lower())).encode('utf8'))
    return stemmed


if __name__ == '__main__':
    if len(sys.argv) != 4:
        print('Usage: python Croatian_stemmer.py input_file output_file stopword_file')
        print('input_file should be an utf8-encoded text file which is then tokenized, ' +
              'stemmed and written in the output_file in a tab-separated fashion.')
        sys.exit(1)

    output_file = open(sys.argv[2], 'w')

    stop_path = sys.argv[3]
    with open(stop_path, 'r') as stop_stream:
        stop = get_stop_words(stop_stream)

    rules = [re.compile(r'^('+osnova+')('+nastavak+r')$') for osnova, nastavak in
            [e.decode('utf8').strip().split(' ') for e in open('rule.txt')]]
    transformations = [e.decode('utf8').strip().split('\t') for e in open('transformations.txt')]

    for token in re.findall(r'\w+', open(sys.argv[1]).read().decode('utf8'), re.UNICODE):
        if token.lower() in stop:
            output_file.write((token+'\t'+token.lower()+'\n').encode('utf8'))
            continue
        output_file.write((token+'\t'+stem_token(transform(token.lower()))+'\n').encode('utf8'))
    output_file.close()

else:
    stem_dir = 'cro_stem'

    resource_package = __name__

    stop_path = '/'.join((stem_dir, 'hrvatski_stoprijeci.txt'))
    rule_path = '/'.join((stem_dir, 'rules.txt'))
    transformation_path = '/'.join((stem_dir, 'transformations.txt'))

    stop_stream = pkg.resource_stream(resource_package, stop_path)
    stop = get_stop_words(stop_stream)
    stop_stream.close()

    rule_stream = pkg.resource_stream(resource_package, rule_path)
    rules = get_rule(rule_stream)
    rule_stream.close()

    transformation_stream = pkg.resource_stream(resource_package, transformation_path)
    transformations = get_transformations(transformation_stream)
    transformation_stream.close()
