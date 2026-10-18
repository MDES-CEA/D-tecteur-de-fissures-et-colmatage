import unittest
from annotation_app.length_analysis import measure, nearby_overlaps


def crack(points, category='open', state='active'):
    return {'geometry': points, 'category': category, 'state': state}


class UniqueLengthTests(unittest.TestCase):
    def test_spatial_bins_partition_corrected_length(self):
        from annotation_app.overlap_measure import distribute_lengths
        result=measure([crack([[0,5],[30,5]]),crack([[5,6],[25,6]])],2)
        zones=[dict(id=1,x=0,y=0,width=20,height=20),dict(id=2,x=20,y=0,width=20,height=20)]
        lengths=distribute_lengths(result['segments'],zones)
        self.assertAlmostEqual(sum(lengths.values()),result['total_px'])
        self.assertAlmostEqual(lengths[1],20)
        self.assertAlmostEqual(lengths[2],10)

    def test_offset_overlap_and_counter_share_same_rule(self):
        lines=[{**crack([[0,0],[100,0]]),'id':1}, {**crack([[30,1],[120,1]]),'id':2}]
        result=measure(lines,tolerance_px=2)
        self.assertAlmostEqual(result['total_px'],120)
        self.assertEqual(result['trace_count'],1)
        self.assertEqual(result['raw_trace_count'],2)
        self.assertEqual(result['overlap_groups'],[[1,2]])
        self.assertAlmostEqual(measure(lines,tolerance_px=0)['total_px'],190)

    def test_crossings_contacts_and_separated_parallel_traces_keep_count(self):
        lines=[{**crack([[0,0],[100,0]]),'id':1}, {**crack([[50,-40],[50,40]]),'id':2},
               {**crack([[100,0],[150,0]]),'id':3}, {**crack([[0,3],[100,3]]),'id':4}]
        result=measure(lines,tolerance_px=2)
        self.assertAlmostEqual(result['total_px'],330)
        self.assertEqual(result['trace_count'],4)

    def test_offset_conflict_partition_and_order_independent(self):
        lines=[{**crack([[0,0],[100,0]]),'id':1}, {**crack([[30,1],[120,1]],'filled'),'id':2}]
        result=measure(lines,tolerance_px=2)
        self.assertAlmostEqual(result['length_px']['conflict'],70)
        self.assertAlmostEqual(result['length_px']['open'],30)
        self.assertAlmostEqual(result['length_px']['filled'],20)
        self.assertEqual(result,measure(lines[::-1],tolerance_px=2))

    def test_a_wobbling_redraw_is_not_deduplicated(self):
        """Limite connue et mesurée du seuil sur la distance perpendiculaire.

        Deux tracés qui suivent la même fissure en oscillant autour l'un de
        l'autre ne sont dédoublonnés que là où ils restent dans la tolérance.
        Sur une vraie fissure redessinée à main levée, le total monte de 40 %.
        Ce test fige le constat ; il n'entérine pas un comportement voulu.
        """
        from math import sin
        base = [[x, 0.0] for x in range(0, 201, 5)]
        wobbling = [[x, 3.0 * sin(x / 7.0)] for x in range(0, 201, 5)]
        result = measure([{**crack(base), 'id': 1}, {**crack(wobbling), 'id': 2}],
                         tolerance_px=2)
        alone = measure([{**crack(base), 'id': 1}], tolerance_px=2)['total_px']
        self.assertGreater(result['total_px'], 1.3 * alone)

    def test_a_reopened_crack_counts_as_sealed(self):
        """Le gel sur les parois prouve un colmatage, meme si la fissure est vide."""
        result = measure([crack([[0, 0], [60, 0]], 'open'),
                          crack([[0, 50], [30, 50]], 'filled'),
                          crack([[0, 90], [10, 90]], 'reopened'),
                          crack([[0, 120], [100, 120]], 'uncertain')])
        self.assertAlmostEqual(result['length_px']['reopened'], 10)
        self.assertAlmostEqual(result['filled_fraction'], 40 / 100)

    def test_diagonal_is_euclidean(self):
        self.assertAlmostEqual(measure([crack([[0,0],[3,4]])])['total_px'], 5)

    def test_partial_reverse_overlap_and_different_nodes(self):
        result = measure([crack([[0,0],[5,0],[10,0]]), crack([[15,0],[3,0]])])
        self.assertAlmostEqual(result['total_px'], 15)
        self.assertAlmostEqual(result['overlap_removed_px'], 7)

    def test_overlap_categories_conflict_not_double_counted(self):
        result = measure([crack([[0,0],[10,0]]), crack([[5,0],[15,0]], 'filled')])
        self.assertEqual(result['length_px']['open'], 5)
        self.assertEqual(result['length_px']['filled'], 5)
        self.assertEqual(result['length_px']['conflict'], 5)
        self.assertEqual(result['filled_fraction'], .5)

    def test_crossing_and_parallel_lines_are_not_collapsed(self):
        result=measure([crack([[0,0],[10,0]]),crack([[5,-5],[5,5]]),crack([[0,1],[10,1]])])
        self.assertAlmostEqual(result['total_px'],30)

    def test_rejected_and_zero_length_are_excluded(self):
        result=measure([crack([[0,0],[10,0]],state='rejected'),crack([[0,0],[0,0]])])
        self.assertEqual(result['total_px'],0)
        self.assertIsNone(result['filled_fraction'])

    def test_order_invariant_and_category_partition(self):
        cracks=[crack([[0,0],[3,4]]),crack([[3,4],[0,0]],'filled')]
        a,b=measure(cracks),measure(cracks[::-1])
        self.assertAlmostEqual(a['total_px'],b['total_px'])
        self.assertAlmostEqual(sum(a['length_px'].values()),a['total_px'])
        self.assertAlmostEqual(sum(a['orientation_length_px']),a['total_px'])

    def test_nearby_lines_are_reviewed_but_not_subtracted(self):
        lines=[{**crack([[0,0],[100,0]]),'id':1}, {**crack([[0,1],[100,1]]),'id':2}]
        self.assertEqual(nearby_overlaps(lines)[0]['trace_ids'],[1,2])
        self.assertEqual(measure(lines)['total_px'],200)
        lines[1]['geometry']=[[0,0],[100,0]]
        self.assertEqual(nearby_overlaps(lines),[])

    def test_empty_ground_truth_reports_false_positive_length(self):
        from types import SimpleNamespace
        import numpy as np
        from outils.evaluate_crack_detection import evaluate_zone
        analysis=SimpleNamespace(scale=1,score=np.zeros((20,20)),
                                 cracks=[{'geometry':[[0,0],[10,0]]}],diagnostics={})
        result=evaluate_zone(SimpleNamespace(analyze=lambda zone:analysis),
                             {'id':1,'x':0,'y':0},[],10)
        self.assertTrue(result['empty_truth'])
        self.assertEqual(result['false_positive_length_px'],10)
        self.assertEqual(result['near_truth'],0)
        self.assertGreater(result['proposed_pixels'],0)


if __name__ == '__main__':
    unittest.main()
