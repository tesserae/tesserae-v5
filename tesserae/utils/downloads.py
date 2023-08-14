import csv
import gzip
import os
import pathlib

from tesserae.utils.exports.highlight import highlight_matches


def _make_row(match, idx, max_score):
    source_txt = highlight_matches(match.source_snippet,
                                   [i[0] for i in match.highlight])
    target_txt = highlight_matches(match.target_snippet,
                                   [i[1] for i in match.highlight])
    features = '; '.join(match.matched_features)
    target_loc = match.target_tag
    source_loc = match.source_tag
    score = match.score
    return {
        'Result': f'{idx}',
        'Target_Loc': f'\"{target_loc}\"',
        'Target_Txt': f'\"{target_txt}\"',
        'Source_Loc': f'\"{source_loc}\"',
        'Source_Txt': f'\"{source_txt}\"',
        'Shared': f'\"{features}\"',
        'Score': f'{score * 10 / max_score}',
        'Raw_Score': f'{score}'
    }


def get_results_filename(search, directory, ext='tsv'):
    """Creates the file path where the search's results are held

    Parameters
    ----------
    search: tesserae.db.entities.Search
    directory: path-like
        Location where results should be stored. If None, a default location
        will be assumed.
    ext: str
        extension, indicating format

    Returns
    -------
    str
        file path where results should be stored
    """
    filename = f'{search.results_id}.{ext}.gz'
    if isinstance(directory, pathlib.PurePath):
        return str(directory / filename)
    return os.path.join(directory, filename)


class ResultsWriter:
    """Writes results to file

    Intended to be used in a context (via "with").
    """

    RESULTS_DIR = os.path.join(os.path.expanduser('~'), 'tess_data', 'results')

    def __init__(self, search, source, target, max_score, ext='tsv'):
        """

        Parameters
        ----------
        search : tesserae.db.entities.Search
            the search whose results are to be written
        source : tesserae.db.entities.Text
            the source text in the search
        target : tesserae.db.entities.Text
            the target text in the search
        max_score : float
            the highest score of all the reuslts in the search
        ext : {'tsv'}
            a file extension indicating the format in which the results are to
            be written
        """
        if not os.path.isdir(ResultsWriter.RESULTS_DIR):
            os.makedirs(ResultsWriter.RESULTS_DIR, exist_ok=True)
        self.filename = get_results_filename(
            search, ext=ext, directory=ResultsWriter.RESULTS_DIR)
        self.search = search
        self.source = source
        self.target = target
        self.max_score = max_score
        self.length = 0

    def record_matches(self, matches):
        """Append matches to results file

        Parameters
        ----------
        matches : List[tesserae.db.entities.Match]
        """
        entries = [
            _make_row(match, self.length + i + 1, self.max_score)
            for i, match in enumerate(matches)
        ]
        self.writer.writerows(entries)
        self.length += len(entries)

    def __enter__(self):
        """Open file for writing and start with commented information"""
        self.fh = gzip.open(self.filename, 'wt', encoding='utf-8', newline='')
        search = self.search
        source = self.source
        target = self.target
        source_title = source.title.lower().replace(" ", "_")
        target_title = target.title.lower().replace(" ", "_")
        # The search parameters and metadata are written as comments to the top
        # of the CSV stream.
        if (search.parameters["method"]["name"] == 'greek_to_latin'):
            comments = [
                '# Tesserae V5 Results',
                '#',
                f'# session         = {search.id}',
                f'# source          = {source.author}.{source_title}',
                f'# target          = {target.author}.{target_title}',
                f'# unit            = {search.parameters["source"]["units"]}',
                f'# stopsize        = {len(search.parameters["method"]["greek_stopwords"])}',
                f'# stbasis         = ',
                f'# greek_stopwords = {search.parameters["method"]["greek_stopwords"]}',
                f'# latin_stopwords = {search.parameters["method"]["latin_stopwords"]}',
                f'# max_dist        = {search.parameters["method"]["max_distance"]}',
                f'# dibasis         = {search.parameters["method"]["distance_basis"]}',
                f'# cutoff          = {0}',
                f'# filter          = off',
            ]
        else: 
            comments = [
                '# Tesserae V5 Results',
                '#',
                f'# session   = {search.id}',
                f'# source    = {source.author}.{source_title}',
                f'# target    = {target.author}.{target_title}',
                f'# unit      = {search.parameters["source"]["units"]}',
                f'# feature   = {search.parameters["method"]["feature"]}',
                f'# stopsize  = {len(search.parameters["method"]["stopwords"])}',
                f'# stbasis   = ',
                f'# stopwords = {search.parameters["method"]["stopwords"]}',
                f'# max_dist  = {search.parameters["method"]["max_distance"]}',
                f'# dibasis   = {search.parameters["method"]["distance_basis"]}',
                f'# cutoff    = {0}',
                f'# filter    = off',
            ]
        self.fh.write('\n'.join(comments))
        self.fh.write('\n')
        self.writer = csv.DictWriter(self.fh, [
            "Result", "Target_Loc", "Target_Txt", "Source_Loc", "Source_Txt",
            "Shared", "Score", "Raw_Score"
        ],
                                     delimiter='\t')
        self.writer.writeheader()
        return self

    def __exit__(self, type, value, traceback):
        self.fh.close()
